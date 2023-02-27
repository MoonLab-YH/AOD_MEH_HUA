import pdb
import torch
import torch.nn as nn
from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import HEADS
from .D_anchor_head import D_AnchorHead
from mmcv.cnn import ConvModule
from torch.distributions.dirichlet import Dirichlet
from mmdet.utils.functions import *
from mmcv.runner import force_fp32
from mmdet.core.export import get_k_for_topk
from mmdet.core.export import add_dummy_nms_for_onnx

global d_value, d_cnt, f_cnt, p_cnt, visOn
d_value, d_cnt, f_cnt, p_cnt, s_cnt, visOn = 0, 0, 0, 0, 0, False

@HEADS.register_module()
class ReWeightSTNet(D_AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.isTrainD = False
        super(ReWeightSTNet, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.D_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.D_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.retina_D = nn.Conv2d(self.feat_channels, self.num_anchors, 3, padding=1)

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=None, proposal_cfg=None, **kwargs):
        if not kwargs['trainD']:
            outs = self.forward(x, **kwargs)
            loss_inputs = outs + (None, gt_bboxes, gt_labels, img_metas)
            losses, head_out = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, **kwargs)
            D_scores, _ = self.forward_D(x, head_out = head_out, **kwargs)
            losses_D = self.loss_D(D_scores, head_out = head_out, **kwargs)
            losses['loss_D'] = losses_D['loss_D']
            return losses, head_out
        else:
            D_scores, _ = self.forward_D(x, **kwargs)
            losses_D = self.loss_D(D_scores, **kwargs)
            return losses_D

    def forward(self, feats, **kwargs):
        return multi_apply(self.forward_single, feats)

    def forward_D(self, feats, **kwargs):
        return multi_apply(self.forward_single_D, feats, **kwargs)

    def forward_single(self, x):
        cls_feat, reg_feat = x, x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        return cls_score, bbox_pred

    def forward_single_D(self, x, **kwargs):
        D_feat = x
        for D_conv in self.D_convs:
            D_feat = D_conv(D_feat)
        d_score = self.retina_D(D_feat)

        return d_score, 0

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, sIdx, num_total_samples, **kwargs):

        prob_threshold, RW1 = 0.5, False
        RW = 'cR1'

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        if kwargs['Labeled'] and (not kwargs['Pseudo']):
            loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples).sum()
            bbox_targets = bbox_targets.reshape(-1, 4)
            bbox_weights = bbox_weights.reshape(-1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            loss_bbox = self.loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples)
        else:
            # loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples).sum()
            # bbox_targets = bbox_targets.reshape(-1, 4)
            # bbox_weights = bbox_weights.reshape(-1, 4)
            # bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            # loss_bbox = self.loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples)
            # loss_cls_real, loss_bbox_real = 0., 0.
            B, C = anchors.size(0), cls_score.size(1)
            loss_cls, loss_bbox = 0., 0.
            label_conf_weights = (cls_score.softmax(dim=-1).max(dim=1)[0] >= prob_threshold)

            global f_cnt, p_cnt, s_cnt, visOn
            if len(label_conf_weights.nonzero()) != 0:
                p_cnt += 1
            f_cnt += 1

            Bbox_preds = bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)
            Cls_scores = cls_score.reshape(B, -1, C)
            Label_conf_weights = label_conf_weights.reshape(B,-1)
            Label_weights = label_weights.reshape(B,-1)
            zipped = zip(Bbox_preds, Cls_scores, Label_conf_weights, anchors, Label_weights)

            iIdx = -1
            for Bbox_pred, Cls_score, Label_conf_weight, anchor, Label_weight in zipped:
                iIdx += 1
                pseudo_pred, pseudo_anchor= Bbox_pred[Label_conf_weight], anchor[Label_conf_weight]
                pseudo_score = Cls_score[Label_conf_weight].softmax(dim=1)
                padding = pseudo_score.new_zeros(pseudo_score.size(0), 1)
                pseudo_score = torch.cat([pseudo_score, padding], dim=-1)
                pseudo_bbox = self.bbox_coder.decode(pseudo_anchor, pseudo_pred)
                nms_bbox, nms_label, nms_idces = multiclass_nms(pseudo_bbox, pseudo_score, prob_threshold,
                                                            {'type':'nms','iou_threshold':0.5}, 100, return_inds=True)
                # NMS_BBOX is pseudo_gt_bboxes..! NMS_LABEL is pseudo_gt_labels..!
                # now that pseudo_gt_bboxes is aquired, we should assign pos_bbox and encode corresponding offsets.
                pseudo_assign_result = self.assigner.assign(anchor, nms_bbox[:,:-1], False, nms_label.to(anchor.device))
                sampling_result = self.sampler.sample(pseudo_assign_result, anchor, nms_bbox[:,:-1])
                pos_inds = sampling_result.pos_inds
                pseudo_targets, pseudo_weights = torch.zeros_like(anchor), torch.zeros_like(anchor)
                pseudo_pos_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
                pseudo_targets[pos_inds, :] = pseudo_pos_targets
                pseudo_weights[pos_inds, :] = 1.0
                loss_bbox += self.loss_bbox(Bbox_pred, pseudo_targets.detach(), pseudo_weights, avg_factor=num_total_samples)
                # generate pseudo labels for classification
                pseudo_labels = labels.new_full(size=(Cls_score.size(0),), fill_value=C)
                pseudo_labels[pos_inds] = nms_label[sampling_result.pos_assigned_gt_inds].to(cls_score.device)
                featSize, padSize =  kwargs['featmap_sizes'][sIdx], kwargs['datas']['img_metas'][iIdx]['pad_shape']
                pseudo_cls_weight = MakeWeights(featSize, padSize, sIdx, anchor.device)
                # Weak Self-Training
                cls_soft = Cls_score.softmax(dim=-1)
                uni = torch.ones_like(Cls_score) / Cls_score.size(-1)
                KLD_cu = F.kl_div(cls_soft.log(), uni, reduction='none').sum(dim=1)
                KLD_uc = F.kl_div(uni.log(), cls_soft, reduction='none').sum(dim=1)
                JSD = 0.5 * (KLD_cu + KLD_uc)
                JSD_flip = JSD.max() - JSD
                JSD_w = (JSD_flip - JSD_flip.min()) / (JSD_flip.max() - JSD_flip.min()+1e-6)
                if RW1:
                    numerator = len(JSD_w)-len(pos_inds)
                    denominator = JSD_w.sum().item() - JSD_w[pos_inds].sum().item()+1e-6
                    scaler = numerator / denominator
                    JSD_w = JSD_w * scaler
                JSD_w[pos_inds] = 1.
                # img = kwargs['datas']['img'].data[iIdx]
                # savedir = 'visualization'
                # visualize(img, f'{savedir}/img.jpg')
                # visualize(JSD_w.reshape(featSize[0], featSize[1], 9).max(dim=2)[0], f'{savedir}/JSD_w_max.jpg')
                # visualize(JSD_w.reshape(featSize[0], featSize[1], 9).mean(dim=2), f'{savedir}/JSD_w_mean.jpg')
                loss_cls += self.loss_cls(Cls_score, pseudo_labels, JSD_w.detach(), avg_factor=num_total_samples).sum()

                # loss_cls & loss_bbox balancing.
                if loss_bbox != 0 and loss_cls != 0:
                    if RW == 'cR1':
                        loss_cls = loss_cls / loss_cls.item() * loss_bbox.item()
                        # loss_cls = loss_cls / loss_cls.item() * loss_cls.item()

                # With Real GT Labels...
                bbox_targets_real = bbox_targets.reshape(B,-1, 4)[iIdx]
                # bbox_weights_real = bbox_weights.reshape(B,-1, 4)[iIdx]
                labels_real = labels.reshape(B,-1)[iIdx]
                # label_weights_real = label_weights.reshape(B,-1)[iIdx]
                # loss_cls_real += self.loss_cls(Cls_score, labels_real, label_weights_real, avg_factor=num_total_samples).sum()
                # loss_bbox_real += self.loss_bbox(Bbox_pred, bbox_targets_real, bbox_weights_real, avg_factor=num_total_samples)
                if f_cnt >= 2000:
                    if sIdx == 0:
                        visOn = True
                    if visOn:
                        if len(pseudo_bbox) == 0:
                            continue
                        decoded_tmp = self.bbox_coder.decode(anchor[pos_inds], pseudo_pos_targets)
                        ori_pred = self.bbox_coder.decode(anchor[pos_inds], Bbox_pred[pos_inds])
                        img = kwargs['datas']['img'].data[iIdx]
                        # savedir = f'tools/visualization{prob_threshold}'
                        savedir = f'visualization_WeakST,p({prob_threshold}) Reweighting'
                        if not os.path.exists(savedir):
                            os.mkdir(savedir)
                        DrawGT(img, pseudo_bbox, f'{savedir}/{s_cnt}_{iIdx}_{sIdx}_beforeNMS.jpg')
                        DrawGT(img, decoded_tmp, f'{savedir}/{s_cnt}_{iIdx}_{sIdx}_cls.jpg', pseudo_labels[pos_inds])
                        GTidx = (labels_real != 20).nonzero().squeeze(dim=1)
                        # DrawGT(img, nms_bbox[:,:-1], f'{savedir}/{s_cnt}_{iIdx}_{sIdx}_afterNMS.jpg')
                        # DrawGT(img, decoded_tmp, f'{savedir}/{s_cnt}_{iIdx}_{sIdx}_posDecoded.jpg')
                        # DrawGT(img, anchor[pos_inds], f'{savedir}/{s_cnt}_{iIdx}_{sIdx}_posAnchor.jpg')
                        # DrawGT(img, ori_pred, f'{savedir}/{s_cnt}_{iIdx}_{sIdx}_oriPred.jpg', pseudo_labels[pos_inds])
                        if len(GTidx != 0):
                            GT_bbox = self.bbox_coder.decode(anchor[GTidx], bbox_targets_real[GTidx])
                            DrawGT(img, GT_bbox, f'{savedir}/{s_cnt}_{iIdx}_{sIdx}_GT.jpg', labels_real[GTidx])
                        if sIdx == 4 and iIdx == B-1:
                            print(f'SSL activation ratio is {p_cnt / f_cnt * 100}%')
                            f_cnt, p_cnt, visOn = 0, 0, False
                            s_cnt = (s_cnt + 1) % 30

        return loss_cls, loss_bbox

    @force_fp32(apply_to=('d_score'))
    def loss_single_D(self, D_score, label_weights, **kwargs):
        label_weights = label_weights.reshape(-1)
        D_score = D_score.permute(0, 2, 3, 1).reshape(-1) + 1e-9
        if kwargs['Labeled']:
            if not kwargs['trainD']:
                # loss_D = F.binary_cross_entropy_with_logits(D_score, torch.zeros_like(D_score), reduce=False)
                loss_D = (F.sigmoid(D_score) - torch.zeros_like(D_score)) ** 2 + 1e-9
                # loss_D = - (1. - F.sigmoid(D_score) + 1e-9).log()
            else:
                # loss_D = F.binary_cross_entropy_with_logits(D_score, torch.ones_like(D_score), reduce=False)
                loss_D = (F.sigmoid(D_score) - torch.ones_like(D_score)) ** 2 + 1e-9
                # loss_D = - (F.sigmoid(D_score) + 1e-9).log()

        else:
            if not kwargs['trainD']:
                # loss_D = F.binary_cross_entropy_with_logits(D_score, torch.ones_like(D_score), reduce=False)
                loss_D = (F.sigmoid(D_score) - torch.ones_like(D_score)) ** 2 + 1e-9
                # loss_D = - (F.sigmoid(D_score) + 1e-9).log()
            else:
                # loss_D = F.binary_cross_entropy_with_logits(D_score, torch.zeros_like(D_score), reduce=False)
                loss_D = (F.sigmoid(D_score) - torch.zeros_like(D_score)) ** 2 + 1e-9
                # loss_D = - (1. - F.sigmoid(D_score) + 1e-9).log()

        loss_D = (loss_D * label_weights).mean()
        # global d_value, d_cnt
        # d_value += F.sigmoid(D_score).mean().item()
        # d_cnt += 1
        # if d_cnt == 2000:
        #     print(f'd_value mean is {d_value / d_cnt}')
        #     d_value, d_cnt = 0, 0

        return loss_D, 0

    @force_fp32(apply_to=('mlvl_cls_scores', 'mlvl_bbox_preds', 'mlvl_anchors'))
    def _get_bboxes(self, mlvl_cls_scores, mlvl_bbox_preds, mlvl_anchors, img_shapes, scale_factors,
                    cfg, rescale=False, with_nms=True, **kwargs):

        cfg = self.test_cfg if cfg is None else cfg
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(mlvl_anchors)
        batch_size = mlvl_cls_scores[0].shape[0]
        nms_pre_tensor = torch.tensor(cfg.get('nms_pre', -1), device=mlvl_cls_scores[0].device, dtype=torch.long)
        mlvl_bboxes, mlvl_scores, mlvl_alphas = [], [], []
        for cls_score, bbox_pred, anchors in zip(mlvl_cls_scores, mlvl_bbox_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels)
            if self.last_activation == 'relu':
                alphas = cls_score.softmax(dim=2)
                S = alphas.sum(dim=2, keepdim=True) + 1e-20
                Smax, _ = S.max(dim=1, keepdim=True)
                gamma = 1
                scores = alphas / ((1-gamma)*Smax + gamma*S + 1e-9)
            else:
                raise NotImplementedError

            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            anchors = anchors.expand_as(bbox_pred)
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if kwargs['isUnc']:
                if kwargs['uPool'] == 'Entropy_ALL':
                    nms_pre = -1
            if nms_pre > 0:
                # Get maximum scores for foreground classes.
                if self.last_activation == 'sigmoid' or self.last_activation == 'relu':
                    max_scores, _ = scores.max(-1)
                elif self.last_activation == 'softmax' or self.last_activation == 'EDL_BG':
                    max_scores, _ = scores[..., :-1].max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(-1, 1).expand_as(topk_inds)
                anchors = anchors[batch_inds, topk_inds, :]
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]
                alphas = alphas[batch_inds, topk_inds, :]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_alphas.append(alphas)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_alphas = torch.cat(mlvl_alphas, dim=1)

        # self.DrawUncertainty(mlvl_anchors, mlvl_cls_scores)
        if torch.onnx.is_in_onnx_export() and with_nms:
            if self.last_activation == 'softmax':
                num_classes = batch_mlvl_scores.shape[2] - 1
                batch_mlvl_scores = batch_mlvl_scores[..., :num_classes]
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores, max_output_boxes_per_class,
                                          iou_threshold, score_threshold, nms_pre, cfg.max_per_img)
        if self.last_activation == 'relu' or self.last_activation == 'sigmoid':
            # Add a dummy background class to the backend when using sigmoid
            padding = batch_mlvl_scores.new_zeros(batch_size, batch_mlvl_scores.shape[1], 1)
            batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes, batch_mlvl_scores):
                det_bbox, det_label, idces = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms,
                                                            cfg.max_per_img, return_inds=True)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [tuple(mlvl_bs) for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)]
        if kwargs['isUnc']:
            scaleUnc = self.ComputeScaleUnc(mlvl_cls_scores)
            AggedUnc = self.AggregateScaleUnc(scaleUnc, type = kwargs['uPool2'])
            return det_results, torch.tensor(AggedUnc)
            # uncertainties = []
            # for Score, Alpha in zip(batch_mlvl_bboxes[...,-1], batch_mlvl_alphas):
            #     if with_nms:
            #         flatten_score = Score.contiguous().view(-1)
            #         nmsIdx = (flatten_score > cfg.score_thr).nonzero()[idces]  # [100,1]
            #         bbIdx, cIdx = (nmsIdx // 20)[:, 0], (nmsIdx % 20)[:, 0]
            #         nms_scores, nms_alphas = Score[bbIdx], Alpha[bbIdx]
            #         dist = Dirichlet(nms_alphas)
            #         samples = dist.sample(torch.tensor([10])) # [sample_size, num_bbox, n_way(20)]
            #         score_for_shannon = nms_scores
            #     else:
            #         dist = Dirichlet(Alpha)
            #         samples = dist.sample(torch.tensor([10])) # [sample_size, num_bbox, n_way(20)]
            #         score_for_shannon = Score
            #     if kwargs['isUnc'] == 'Shannon':
            #         entropy = (-score_for_shannon * score_for_shannon.log()).sum(dim=1)
            #         uncertainties.append(entropy)
            #     elif kwargs['isUnc'] == 'Aleatoric':
            #         ent = (-samples*samples.log()).sum(dim=-1)
            #         aleatoric = ent.mean(dim=0)
            #         uncertainties.append(aleatoric)
            #     elif kwargs['isUnc'] == 'Epistemic':
            #         avg = samples.mean(dim=0)
            #         total = (-avg * avg.log()).sum(dim=1)
            #         ent = (-samples * samples.log()).sum(dim=-1)
            #         aleatoric = ent.mean(dim=0)
            #         epistemic = total - aleatoric
            #         uncertainties.append(epistemic)
            #     elif kwargs['isUnc'] == 'Total':
            #         avg = samples.mean(dim=0)
            #         total = (-avg*avg.log()).sum(dim=1)
            #         uncertainties.append(total)
            # return det_results, torch.stack(uncertainties)
        else:
            return det_results

    def simple_test(self, feats, img_metas, rescale=False, **kwargs):
        # input : self.bbox_head.simple_test(feat, img_metas, rescale=rescale, **kwargs)
        outs = self.forward(feats)
        if not kwargs['isEval'] and kwargs['uPool'] == 'Entropy_NoNMS':
            results_list = self.get_bboxes(*outs, img_metas, rescale=rescale, with_nms=False, **kwargs)
        elif not kwargs['isEval'] and kwargs['uPool'] == 'Entropy_ALL':
            results_list = self.get_bboxes(*outs, img_metas, rescale=rescale, with_nms=False, **kwargs)
        else:
            results_list = self.get_bboxes(*outs, img_metas, rescale=rescale, **kwargs)

        if not kwargs['isEval']:
            real_results_list = results_list[0]
            uncertainties = results_list[1]
            return real_results_list, uncertainties
        else:
            return results_list

    def DrawUncertainty(self,  mlvl_cls_scores):
        _, oriH, oriW = mlvl_cls_scores[0][0].shape
        print('\n')
        for sIdx, slvl_scores in enumerate(mlvl_cls_scores):
            for iIdx, simg_scores in enumerate(slvl_scores):
                # unc = uncertainty[sIdx*1000:(sIdx+1)*1000]
                C, H, W = simg_scores.shape
                simg_scores = simg_scores.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
                slvl_alphas = simg_scores.softmax(dim=1) # [ N, 20]
                slvl_maxconf = slvl_alphas.max(dim=1)[0]
                FGIdx, BGIdx = (slvl_maxconf > 0.3), (slvl_maxconf <= 0.3)
                if not True in FGIdx:
                    print(f'img{iIdx} scale{sIdx} All filtered out')
                    continue

                dist = Dirichlet(slvl_alphas)
                samples = dist.sample(torch.tensor([10]))  # [sample_size, num_bbox, n_way(20)]
                avg = samples.mean(dim=0)
                total = (-avg * avg.log()).sum(dim=1)
                ent = (-samples * samples.log()).sum(dim=-1)
                aleatoric = ent.mean(dim=0)
                epistemic = total - aleatoric
                epicp = epistemic.clone()
                alecp = aleatoric.clone()
                epistemic[BGIdx] = 0
                aleatoric[BGIdx] = 0
                avg_epistemic = epistemic.reshape(H,W,9).mean(dim=2) # [H,W]
                visualize(avg_epistemic, f'visualization/{iIdx}img_{sIdx}scale_avgEpi.jpg', size=(oriH,oriW), heatmap=True)
                max_epistemic = epistemic.reshape(H, W, 9).max(dim=2)[0]  # [H,W]
                visualize(max_epistemic, f'visualization/{iIdx}img_{sIdx}scale_maxEpi.jpg', size=(oriH,oriW), heatmap=True)
                avg_aleatoric = aleatoric.reshape(H, W, 9).mean(dim=2)  # [H,W]
                visualize(avg_aleatoric, f'visualization/{iIdx}img_{sIdx}scale_avgAle.jpg', size=(oriH, oriW), heatmap=True)
                max_aleatoric = aleatoric.reshape(H, W, 9).max(dim=2)[0]  # [H,W]
                visualize(max_aleatoric, f'visualization/{iIdx}img_{sIdx}scale_maxAle.jpg', size=(oriH, oriW), heatmap=True)
                maxAlpha = slvl_alphas.max(dim=1)[0]
                maxmaxAlpha = maxAlpha.reshape(H, W, 9).max(dim=2)[0]
                visualize(maxmaxAlpha, f'visualization/{iIdx}img_{sIdx}scale_maxmaxAlpha.jpg', size=(oriH, oriW), heatmap=True)
                avgmaxAlpha = maxAlpha.reshape(H, W, 9).mean(dim=2)
                visualize(avgmaxAlpha, f'visualization/{iIdx}img_{sIdx}scale_avgmaxAlpha.jpg', size=(oriH, oriW), heatmap=True)
                selfWmaxEpi = maxmaxAlpha * max_epistemic
                visualize(selfWmaxEpi, f'visualization/{iIdx}img_{sIdx}scale_selfWmaxEpi.jpg', size=(oriH, oriW), heatmap=True)
                selfWavgEpi = maxmaxAlpha * avg_epistemic
                visualize(selfWavgEpi, f'visualization/{iIdx}img_{sIdx}scale_selfWavgEpi.jpg', size=(oriH, oriW), heatmap=True)
                selfWavgEpi = (maxmaxAlpha + 1) * avg_epistemic
                visualize(selfWavgEpi, f'visualization/{iIdx}img_{sIdx}scale_PlusselfWavgEpi.jpg', size=(oriH, oriW), heatmap=True)

    def ComputeScaleUnc(self, mlvl_cls_scores):
        S = len(mlvl_cls_scores)
        B, _, oriH, oriW = mlvl_cls_scores[0].shape
        output = [[{} for _ in range(S)] for _ in range(B)]
        for sIdx, slvl_scores in enumerate(mlvl_cls_scores):
            for iIdx, simg_scores in enumerate(slvl_scores):
                simg_scores = simg_scores.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
                slvl_alphas = simg_scores.softmax(dim=1)  # [ N, 20]
                slvl_maxconf = slvl_alphas.max(dim=1)[0]
                FGIdx, BGIdx = (slvl_maxconf > 0.3), (slvl_maxconf <= 0.3)
                if not True in FGIdx:
                    continue
                fil_alphas = slvl_alphas[FGIdx]
                fil_dist = Dirichlet(fil_alphas)
                samples = fil_dist.sample(torch.tensor([1000]))  # [sample_size, num_bbox, n_way(20)]
                avg = samples.mean(dim=0)
                total = (-avg * avg.log()).sum(dim=1)
                ent = (-samples * samples.log()).sum(dim=-1)
                aleatoric = ent.mean(dim=0)
                epistemic = total - aleatoric
                classes = slvl_alphas[FGIdx].argmax(dim=1).unique()
                for cls in classes:
                    clsIdx = slvl_alphas[FGIdx].argmax(dim=1) == cls
                    clsEpi = epistemic[clsIdx].mean()
                    clsAle = aleatoric[clsIdx].mean()
                    output[iIdx][sIdx][f'{cls}'] = (clsAle, clsEpi)
        return output

    def AggregateScaleUnc(self, scaleUnc, type):
        output = []
        if type == 'scaleAvg_classAvg':
            for iIdx, imgUnc in enumerate(scaleUnc):
                Uncs = []
                for sIdx, sUncs in enumerate(imgUnc):
                    Uncs2 = []
                    for (cls, (Ale,Epi)) in sUncs.items():
                        Uncs2.append(Epi.item())
                    if Uncs2:
                        Uncs.append(np.array(Uncs2).mean())
                if Uncs:
                    Unc = np.array(Uncs).mean()
                    output.append(Unc)
                else:
                    output.append(0)
        elif type == 'scaleSum_classSum':
            for iIdx, imgUnc in enumerate(scaleUnc):
                Uncs = []
                for sIdx, sUncs in enumerate(imgUnc):
                    for (cls, (Ale, Epi)) in sUncs.items():
                        Uncs.append(Epi.item())
                if Uncs:
                    Unc = np.array(Uncs).sum()
                    output.append(Unc)
                else:
                    output.append(0)
        elif type == 'scaleSum_classAvg':
            for iIdx, imgUnc in enumerate(scaleUnc):
                Uncs = []
                for sIdx, sUncs in enumerate(imgUnc):
                    Uncs2 = []
                    for (cls, (Ale, Epi)) in sUncs.items():
                        Uncs2.append(Epi.item())
                    if Uncs2:
                        Uncs.append(np.array(Uncs2).mean())
                if Uncs:
                    Unc = np.array(Uncs).sum()
                    output.append(Unc)
                else:
                    output.append(0)
        elif type == 'scaleAvg_classSum':
            for iIdx, imgUnc in enumerate(scaleUnc):
                Uncs = []
                for sIdx, sUncs in enumerate(imgUnc):
                    Uncs2 = []
                    for (cls, (Ale, Epi)) in sUncs.items():
                        Uncs2.append(Epi.item())
                    if Uncs2:
                        Uncs.append(np.array(Uncs2).sum())
                if Uncs:
                    Unc = np.array(Uncs).mean()
                    output.append(Unc)
                else:
                    output.append(0)
        return output

    def DrawSingleConf(self,  cls_scores, sIdx):
        _, oriH, oriW = cls_scores[0].shape
        print('\n')
        for iIdx, simg_scores in enumerate(cls_scores):
            simg_scores = simg_scores.permute(1, 2, 0).reshape(-1, self.cls_out_channels) # [H,W,9,20] --> [N, 20]
            slvl_alphas = simg_scores.softmax(dim=1) # [ N, 20]
            # slvl_alphas[slvl_alphas <= 0.3] = 0
            slvl_anchors = slvl_alphas.reshape(oriH, oriW, 9, self.cls_out_channels)
            slvl_maxanc = slvl_anchors.max(dim=2)[0] # [H,W,20]
            slvl_maxcls = slvl_maxanc.max(dim=2)[0] # [H,W]
            visualize(slvl_maxcls, f'visualization/{iIdx+1}_img_{sIdx}_conf.jpg', size=(512,512), heatmap=True)



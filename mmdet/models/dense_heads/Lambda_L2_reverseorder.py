import pdb
import torch
import torch.nn as nn
from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import HEADS
from .L_anchor_head import L_AnchorHead
from mmcv.cnn import ConvModule
from torch.distributions.dirichlet import Dirichlet
from mmdet.utils.functions import *
from mmcv.runner import force_fp32
from mmdet.core.export import get_k_for_topk
from mmdet.core.export import add_dummy_nms_for_onnx

global d_value, d_cnt, f_cnt, p_cnt, visOn
d_value, d_cnt, f_cnt, p_cnt, s_cnt, visOn = 0, 0, 0, 0, 0, False

@HEADS.register_module()
class Lambda_L2Net_reverse(L_AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(type='AnchorGenerator', octave_base_scale=4, scales_per_octave=3, ratios=[0.5, 1.0, 2.0], strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01,
                               override=dict(type='Normal', name='retina_cls', std=0.01, bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.isTrainD = False
        super(Lambda_L2Net_reverse, self).__init__(num_classes, in_channels, anchor_generator=anchor_generator, init_cfg=init_cfg, **kwargs)
        self.L_names = ['retina_L', 'L_convs']

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.L_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                             conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            self.reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                             conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            self.L_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                           conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(self.feat_channels, self.num_anchors * self.cls_out_channels, 3, padding=1)
        self.retina_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.retina_L = nn.Conv2d(self.feat_channels, self.num_anchors, 3, padding=1)

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=None, proposal_cfg=None, **kwargs):
        outs = self.forward(x, **kwargs)
        loss_inputs = outs + (None, gt_bboxes, gt_labels, img_metas)
        losses, head_out = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, **kwargs)
        return losses, head_out

    def forward_train_L(self, prev_loss, head_out, x, **kwargs):
        L_scores = self.forward_L(x, head_out, **kwargs)
        losses = self.loss_L(L_scores, head_out, prev_loss, **kwargs)

        def vis_img_lscore_loss():
            img = kwargs['_data']['img'].data[0][0]
            _,H,W = img.shape
            visualize(img, f'img.jpg')
            for sl_loss, sl_l, sl_feat, sIdx in zip(prev_loss, L_scores, x, range(5)):
                _,_,h,w = sl_feat.shape
                loss0 = sl_loss.reshape(2, h, w, 9)[0].sum(dim=-1)
                lambda0 = sl_l[0].permute([1, 2, 0]).sum(dim=-1)
                visualize(loss0, f'loss{sIdx}.jpg', color=False, size=(H,W))
                visualize(lambda0, f'lambda{sIdx}.jpg', color=False, size=(H,W))

        return losses

    def forward(self, feats, **kwargs):
        return multi_apply(self.forward_single, feats)

    def forward_L(self, feats, head_out, **kwargs):
        return multi_apply(self.forward_single_L, feats)[0]

    def forward_single(self, x):
        cls_feat, reg_feat = x, x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        return cls_score, bbox_pred

    def forward_single_L(self, x):
        L_feat = x
        for L_conv in self.L_convs:
            L_feat = L_conv(L_feat)
        L_score = self.retina_L(L_feat)
        L_score = self.relu(L_score)
        # L_score = F.sigmoid(L_score)*5
        return L_score, 0

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, sIdx, num_total_samples, **kwargs):

        prob_threshold, RW1 = 0.5, False
        RW = 'cR1'

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        if kwargs['Labeled'] and (not kwargs['Pseudo']):
            loss_noR = self.loss_cls(cls_score, labels, reduction_override='none').sum(dim=-1)
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
            f_cnt += 1

            Bbox_preds = bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)
            Cls_scores = cls_score.reshape(B, -1, C)
            Label_conf_weights = label_conf_weights.reshape(B,-1)
            Label_weights = label_weights.reshape(B,-1)
            zipped = zip(Bbox_preds, Cls_scores, Label_conf_weights, anchors, Label_weights)
            losses_noR = []

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
                losses_noR.append(self.loss_cls(Cls_score, pseudo_labels, reduction_override='none').sum(dim=-1))
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
            loss_noR = torch.stack(losses_noR).flatten()
        return loss_cls, loss_bbox, loss_noR

    @force_fp32(apply_to=('L_score'))
    def loss_single_L(self, L_score, loss, label_weights, bbox_weights, **kwargs):
        weights = bbox_weights[...,0] # Or it can be just label_weights. (It was..! for 72.10)
        weights = weights.reshape(-1)
        L_score2 = L_score.permute(0, 2, 3, 1).reshape(-1) + 1e-9
        loss_L = torch.abs(L_score2 - loss)
        loss_L = (loss_L * weights).pow(2).mean()*5

        def visualize_lossL_info():
            B,C,H,W = L_score.shape
            b,_,h,w = kwargs['_data']['img'].data[0].shape
            for i in range(B):
                visualize(kwargs['_data']['img'].data[0][i], f'visualization/img_{i}.jpg')
                visualize(L_score.permute(0,2,3,1)[i].sum(dim=-1), f'visualization/L_{i}.jpg', size=(h,w), heatmap=True)
                visualize(label_weights.reshape(B,H,W,C)[i].sum(dim=-1), f'visualization/weight_{i}.jpg', size=(h,w), heatmap=True)
                visualize(loss.reshape(B,H,W,C)[i].sum(dim=-1), f'visualization/prevLoss_{i}.jpg', size=(h,w), heatmap=True)
        # visualize_lossL_info()
        return loss_L, 0

    @force_fp32(apply_to=('mlvl_cls_scores', 'mlvl_bbox_preds', 'mlvl_anchors'))
    def _get_bboxes(self, mlvl_cls_scores, mlvl_bbox_preds, mlvl_anchors, img_shapes, scale_factors,
                    cfg, rescale=False, with_nms=True, **kwargs):

        cfg = self.test_cfg if cfg is None else cfg
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(mlvl_anchors)
        batch_size = mlvl_cls_scores[0].shape[0]
        nms_pre_tensor = torch.tensor(cfg.get('nms_pre', -1), device=mlvl_cls_scores[0].device, dtype=torch.long)
        L_scores = kwargs['L_scores'] if 'L_scores' in kwargs else mlvl_cls_scores
        mlvl_bboxes, mlvl_scores, mlvl_alphas, mlvl_Ls, mlvl_idces = [], [], [], [], []
        for cls_score, bbox_pred, anchors, l_scores in zip(mlvl_cls_scores, mlvl_bbox_preds, mlvl_anchors, L_scores):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels)
            l_scores = l_scores.permute(0,2,3,1).reshape(batch_size, -1)
            if self.last_activation == 'relu':
                alphas = cls_score.softmax(dim=2)
                S = alphas.sum(dim=2, keepdim=True) + 1e-20
                Smax, _ = S.max(dim=1, keepdim=True)
                gamma = 1
                scores = alphas / ((1-gamma)*Smax + gamma*S + 1e-9)
                Idces = torch.arange(alphas.size(1)).expand_as(alphas[..., 0])
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
                Idces = Idces[batch_inds, topk_inds]
                anchors = anchors[batch_inds, topk_inds, :]
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]
                alphas = alphas[batch_inds, topk_inds, :]
                l_scores = l_scores[batch_inds, topk_inds]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_alphas.append(alphas)
            mlvl_idces.append(Idces)
            mlvl_Ls.append(l_scores)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        bak_mlvl_scores = mlvl_scores

        # self.DrawUncertainty(mlvl_anchors, mlvl_cls_scores)
        if torch.onnx.is_in_onnx_export() and with_nms:
            if self.last_activation == 'softmax':
                num_classes = batch_mlvl_scores.shape[2] - 1
                batch_mlvl_scores = batch_mlvl_scores[..., :num_classes]
            max_output_boxes_per_class = cfg.nms.get('max_output_boxes_per_class', 200)
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
            det_results, pos_bboxes = [], []
            for j, (mlvl_bboxes, mlvl_scores) in enumerate(zip(batch_mlvl_bboxes, batch_mlvl_scores)):
                det_bbox, det_label, idces = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms,
                                                            cfg.max_per_img, return_inds=True)
                det_results.append(tuple([det_bbox, det_label]))
                def ShowNMSed(det_bbox, det_label, idces, iIdx, show_thr, name = 'tmp', **kwargs):
                    h, w = kwargs['_meta'][iIdx]['img_shape'][:-1]
                    img_show = kwargs['_data'][:, :, :h, :w]
                    ori_h, ori_w = kwargs['_meta'][iIdx]['ori_shape'][:-1]
                    img_show = F.interpolate(img_show, size=(ori_h, ori_w))[iIdx]
                    show_idces = det_bbox[:,-1] > show_thr
                    thr_bbox = det_bbox[show_idces][:,:4]
                    thr_label = det_label[show_idces]
                    DrawGT(img_show, thr_bbox, f'visualization2/{name}.jpg', thr_label)
                def GetObjectIdx(det_bbox, mlvl_bboxes, score_thr, iou_thr):
                    filtered_bbox = det_bbox[det_bbox[:,-1] > score_thr]
                    overlaps = self.assigner.iou_calculator(mlvl_bboxes, filtered_bbox)
                    overlaps_idx = overlaps > iou_thr
                    return overlaps_idx
                if kwargs['isUnc'] and kwargs['uPool'] == 'Entropy_NMS':
                    pos_bboxes.append(GetObjectIdx(det_bbox, mlvl_bboxes, score_thr=0.3, iou_thr=0.5))
                if 'showNMS' in kwargs and kwargs['showNMS']:
                    imgName = kwargs['batchIdx']*len(batch_mlvl_bboxes) + j
                    ShowNMSed(det_bbox, det_label, idces, j, show_thr=0.3, name=imgName, **kwargs)
        else:
            det_results = [tuple(mlvl_bs) for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)]
        if kwargs['isUnc']:
            L_scores = kwargs['L_scores'] if 'L_scores' in kwargs else None
            if kwargs['uPool'] == 'Entropy_NMS':
                # isSuccess = self.AssertIdces(mlvl_cls_scores, bak_mlvl_scores, mlvl_idces)
                objUnc = self.ComputeObjUnc(mlvl_cls_scores, pos_bboxes, bak_mlvl_scores, mlvl_Ls, mlvl_idces)
                AggedUnc = self.AggregateObjScaleUnc(objUnc, type=kwargs['uPool2'], **kwargs)
            else:
                scaleUnc = self.ComputeScaleUnc(mlvl_cls_scores, L_scores)
                AggedUnc = self.AggregateScaleUnc(scaleUnc, type = kwargs['uPool2'])
            # print(f'L_score is {[i.max().item() for i in L_scores]}')
            if 'draw' in kwargs and kwargs['draw']:
                self.DrawUncertainty(mlvl_cls_scores, L_scores, **kwargs)
            if 'saveUnc' in kwargs and kwargs['saveUnc']:
                if 'showMaxConf' in kwargs and kwargs['showMaxConf']:
                    maxconf, maxes = getMaxConf(mlvl_cls_scores, self.cls_out_channels)
                    self.SaveUnc(AggedUnc, maxconf = maxconf, **kwargs)
                else:
                    self.SaveUnc(AggedUnc, **kwargs)
            if 'saveMaxConf' in kwargs and kwargs['saveMaxConf']:
                maxconf, maxes = getMaxConf(mlvl_cls_scores, self.cls_out_channels)
            if 'scaleUnc' in kwargs and kwargs['scaleUnc']:
                return det_results, AggedUnc, scaleUnc
            elif 'saveMaxConf' in kwargs and kwargs['saveMaxConf']:
                return det_results, AggedUnc, maxconf
            else:
                return det_results, AggedUnc
        else:
            return det_results

    def AssertIdces(self, mlvl_cls_scores, mlvl_scores, mlvl_idces):
        for cls_scores, scores, idces in zip(mlvl_cls_scores, mlvl_scores, mlvl_idces):
            B = len(cls_scores)
            cls_scores = cls_scores.permute(0,2,3,1).reshape(B, -1, self.cls_out_channels)
            for cls_score, score, idx in zip(cls_scores, scores, idces):
                idxed = cls_score[idx]
                matched = idxed == score
                if not matched.all():
                    return False
        return True


    def simple_test(self, feats, img_metas, rescale=False, **kwargs):
        # input : self.bbox_head.simple_test(feat, img_metas, rescale=rescale, **kwargs)
        outs = self.forward(feats)
        L_scores = self.forward_L(feats, head_out=None)
        # if 'justOut' in kwargs and kwargs['justOut']:
        #     return outs, L_scores
        if not kwargs['isEval'] and kwargs['uPool'] == 'Entropy_NoNMS':
            results_list = self.get_bboxes(*outs, img_metas, rescale=rescale, with_nms=False, **kwargs)
        elif not kwargs['isEval'] and kwargs['uPool'] == 'Entropy_ALL':
            withNMS = True if kwargs['showNMS'] else False
            results_list = self.get_bboxes(*outs, img_metas, rescale=rescale, with_nms=withNMS, L_scores = L_scores, **kwargs)
        elif not kwargs['isEval'] and kwargs['uPool'] == 'Entropy_NMS':
            results_list = self.get_bboxes(*outs, img_metas, rescale=rescale, with_nms=True, L_scores = L_scores, **kwargs)
        else:
            results_list = self.get_bboxes(*outs, img_metas, rescale=rescale, **kwargs)

        if not kwargs['isEval']:
            real_results_list = results_list[0]
            uncertainties = results_list[1:]
            if 'scaleUnc' in kwargs and kwargs['scaleUnc']: return results_list
            return (real_results_list, *uncertainties)
        else:
            return results_list

    def SaveUnc(self, AggedUnc, **kwargs):
        for iIdx, aggedUnc in zip(range(len(AggedUnc)), AggedUnc):
            if 'name' in kwargs: name = kwargs['name']
            else: name = iIdx
            if 'dirName' in kwargs: dirName = os.path.join(f'visualization/{kwargs["dirName"]}')
            else: dirName = 'visualization'
            if 'maxconf' in kwargs:
                maxconf = kwargs['maxconf']
                visualize(kwargs['_data'][iIdx], f'{dirName}/{name}img_{aggedUnc}_max{maxconf}.jpg')
            else:
                visualize(kwargs['_data'][iIdx], f'{dirName}/{name}img_{aggedUnc}.jpg')

    def DrawUncertainty(self,  mlvl_cls_scores, mlvl_l_scores, **kwargs):
        _, oriH, oriW = mlvl_cls_scores[0][0].shape
        oriH, oriW = 8*oriH, 8*oriW
        print('\n')
        for sIdx, slvl_scores in enumerate(mlvl_cls_scores):
            for iIdx, simg_scores in enumerate(slvl_scores):
                # unc = uncertainty[sIdx*1000:(sIdx+1)*1000]
                if 'name' in kwargs: name = kwargs['name']
                else: name = iIdx
                C, H, W = simg_scores.shape
                simg_scores = simg_scores.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
                slvl_alphas = simg_scores.softmax(dim=1) # [ N, 20]
                slvl_maxconf = slvl_alphas.max(dim=1)[0]
                FGIdx, BGIdx = (slvl_maxconf > 0.3), (slvl_maxconf <= 0.3)
                visualize(kwargs['_data'][iIdx], f'visualization/{name}img.jpg')
                if not True in FGIdx:
                    print(f'img{iIdx} scale{sIdx} All filtered out')
                    continue
                maxAlpha = slvl_alphas.max(dim=1)[0]
                maxmaxAlpha = maxAlpha.reshape(H, W, 9).max(dim=2)[0]
                visualize(maxmaxAlpha, f'visualization/{name}img_{sIdx}scale_Alpha.jpg', size=(oriH, oriW), heatmap=True)
                l_scores = mlvl_l_scores[sIdx][iIdx].permute(1, 2, 0).reshape(-1, 1)
                # # TODO CHANGE IT LATER !! #
                l_scores[l_scores<= 0.002] = 0.002
                # # TODO CHANGE IT LATER !! #
                ##################### Be careful of eps!! #####################
                # eps = l_scores.mean() * 0.01
                eps = 1e-9
                l_scores = l_scores.mean() / (l_scores + eps) * 25
                ###############################################################
                l_vis = l_scores.reshape(H, W, 9).sum(dim=-1)
                slvl_L_alphas = slvl_alphas * l_scores
                visualize(l_vis, f'visualization/{name}img_{sIdx}scale__Lambda_{l_vis.max()}.jpg',
                          size=(oriH, oriW), heatmap=True)
                for alphas, _type in zip([slvl_alphas, slvl_L_alphas], ['_NOL_','_L_']):
                    dist = Dirichlet(alphas)
                    samples = dist.sample(torch.tensor([50]))  # [sample_size, num_bbox, n_way(20)]
                    avg = samples.mean(dim=0)
                    total = (-avg * avg.log()).sum(dim=1)
                    ent = (-samples * samples.log()).sum(dim=-1)
                    aleatoric = ent.mean(dim=0)
                    epistemic = total - aleatoric
                    epistemic[BGIdx] = 0
                    aleatoric[BGIdx] = 0
                    avg_epistemic = epistemic.reshape(H,W,9).mean(dim=2) # [H,W]
                    visualize(avg_epistemic, f'visualization/{name}img_{sIdx}scale{_type}avgEpi.jpg',
                              size=(oriH, oriW), heatmap=True)
                    max_epistemic = epistemic.reshape(H, W, 9).max(dim=2)[0]  # [H,W]
                    visualize(max_epistemic, f'visualization/{name}img_{sIdx}scale{_type}maxEpi.jpg',
                              size=(oriH, oriW), heatmap=True)
                # avg_aleatoric = aleatoric.reshape(H, W, 9).mean(dim=2)  # [H,W]
                # visualize(avg_aleatoric, f'visualization/{iIdx}img_{sIdx}scale_avgAle.jpg', size=(oriH, oriW), heatmap=True)
                # max_aleatoric = aleatoric.reshape(H, W, 9).max(dim=2)[0]  # [H,W]
                # visualize(max_aleatoric, f'visualization/{iIdx}img_{sIdx}scale_maxAle.jpg', size=(oriH, oriW), heatmap=True)

    def ComputeObjUnc(self, mlvl_cls_scores, pos_bboxes, mlvl_scores, mlvl_Ls, mlvl_idces):
        S = len(mlvl_cls_scores)
        B, _, oriH, oriW = mlvl_cls_scores[0].shape
        lenObj = [i.size(1) for i in pos_bboxes]
        output = [[[{} for _ in range(S)] for _ in range(lenObj[b])] for b in range(B)]
        for sIdx, slvl_scores in enumerate(mlvl_cls_scores):
            for iIdx, simg_scores in enumerate(slvl_scores):
                simg_scores = simg_scores.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
                slvl_alphas = simg_scores.softmax(dim=1)  # [ N, 20]
                slvl_maxconf = slvl_alphas.max(dim=1)[0]
                FGIdx, BGIdx = (slvl_maxconf > 0.3), (slvl_maxconf <= 0.3)
                if not True in FGIdx:
                    continue
                topk_score = mlvl_scores[sIdx][iIdx]
                start, end = StartEnd(mlvl_scores, sIdx)
                pos_bbox = pos_bboxes[iIdx][start:end]
                if len(pos_bbox.nonzero()) == 0:
                    continue
                FG_pos_bbox = pos_bbox & (topk_score.max(dim=1)[0] > 0.3)[:,None].expand_as(pos_bbox)
                posIdx, objInfo = FG_pos_bbox.nonzero()[:, 0], FG_pos_bbox.nonzero()[:, 1]
                if len(posIdx) == 0:
                    continue
                pos_scores = topk_score[posIdx]
                pos_l_scores = mlvl_Ls[sIdx][iIdx][posIdx]
                eps = 1e-7
                pos_l_scores = pos_l_scores.mean() / (pos_l_scores + eps) * 25
                # pos_alphas = pos_scores.softmax(dim=1)  # [ N, 20]
                pos_alphas = pos_scores
                pos_alphas = pos_alphas * pos_l_scores[:,None]
                pos_dist = Dirichlet(pos_alphas)
                samples = pos_dist.sample(torch.tensor([500]))  # [sample_size, num_bbox, n_way(20)]
                avg = samples.mean(dim=0)
                total = (-avg * avg.log()).sum(dim=1)
                ent = (-samples * samples.log()).sum(dim=-1)
                aleatoric = ent.mean(dim=0)
                epistemic = total - aleatoric
                pos_cls = pos_scores.argmax(dim=1)
                for obj in objInfo.unique():
                    objIdx = objInfo == obj
                    obj_score = pos_scores[objIdx]
                    classes = obj_score.argmax(dim=1).unique()
                    for cls in classes:
                        clsIdx = pos_cls == cls
                        objClsIdx = objIdx & clsIdx
                        objClsEpi = epistemic[objClsIdx].mean()
                        objClsAle = aleatoric[objClsIdx].mean()
                        output[iIdx][obj][sIdx][f'{cls}'] = (objClsAle, objClsEpi)
        return output

    def ComputeScaleUnc(self, mlvl_cls_scores, L_scores):
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
                l_scores = L_scores[sIdx][iIdx].permute(1, 2, 0).reshape(-1, 1)
                eps = 1e-7
                l_scores = l_scores.mean() / (l_scores + eps) * 25
                slvl_alphas = slvl_alphas * l_scores
                fil_alphas = slvl_alphas[FGIdx]
                fil_dist = Dirichlet(fil_alphas)
                samples = fil_dist.sample(torch.tensor([500]))  # [sample_size, num_bbox, n_way(20)]
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

    def AggregateObjScaleUnc(self, objScaleClsUnc, type, clsW = False, **kwargs):
        output = []
        aggFuncs = ExtractAggFunc(type)
        for iIdx, imgUnc in enumerate(objScaleClsUnc):
            uncObjs, clsImg = [], {}
            for objIdx, objUnc in enumerate(imgUnc):
                uncScales = []
                for sIdx, sUncs in enumerate(objUnc):
                    uncClss = []
                    for (cls, (Ale, Epi)) in sUncs.items():
                        uncClss.append(Epi.item())
                        clsImg[cls] = ''
                    if uncClss:
                        uncScales.append(aggFuncs['class'](torch.tensor(uncClss)))
                if uncScales:
                    uncObjs.append(aggFuncs['scale'](torch.tensor(uncScales)))
            if uncObjs:
                output.append(aggFuncs['object'](torch.tensor(uncObjs)).item())
            else:
                output.append(0)
            if clsW:
                output[-1] *= len(clsImg)
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
    # def DrawPrevUncertainty(self,  mlvl_cls_scores):
    #     _, oriH, oriW = mlvl_cls_scores[0][0].shape
    #     print('\n')
    #     for sIdx, slvl_scores in enumerate(mlvl_cls_scores):
    #         for iIdx, simg_scores in enumerate(slvl_scores):
    #             # unc = uncertainty[sIdx*1000:(sIdx+1)*1000]
    #             C, H, W = simg_scores.shape
    #             simg_scores = simg_scores.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
    #             slvl_alphas = simg_scores.softmax(dim=1) # [ N, 20]
    #             slvl_maxconf = slvl_alphas.max(dim=1)[0]
    #             FGIdx, BGIdx = (slvl_maxconf > 0.3), (slvl_maxconf <= 0.3)
    #             if not True in FGIdx:
    #                 print(f'img{iIdx} scale{sIdx} All filtered out')
    #                 continue
    #
    #             dist = Dirichlet(slvl_alphas)
    #             samples = dist.sample(torch.tensor([10]))  # [sample_size, num_bbox, n_way(20)]
    #             avg = samples.mean(dim=0)
    #             total = (-avg * avg.log()).sum(dim=1)
    #             ent = (-samples * samples.log()).sum(dim=-1)
    #             aleatoric = ent.mean(dim=0)
    #             epistemic = total - aleatoric
    #             epicp = epistemic.clone()
    #             alecp = aleatoric.clone()
    #             epistemic[BGIdx] = 0
    #             aleatoric[BGIdx] = 0
    #             avg_epistemic = epistemic.reshape(H,W,9).mean(dim=2) # [H,W]
    #             visualize(avg_epistemic, f'visualization/{iIdx}img_{sIdx}scale_avgEpi.jpg', size=(oriH,oriW), heatmap=True)
    #             max_epistemic = epistemic.reshape(H, W, 9).max(dim=2)[0]  # [H,W]
    #             visualize(max_epistemic, f'visualization/{iIdx}img_{sIdx}scale_maxEpi.jpg', size=(oriH,oriW), heatmap=True)
    #             avg_aleatoric = aleatoric.reshape(H, W, 9).mean(dim=2)  # [H,W]
    #             visualize(avg_aleatoric, f'visualization/{iIdx}img_{sIdx}scale_avgAle.jpg', size=(oriH, oriW), heatmap=True)
    #             max_aleatoric = aleatoric.reshape(H, W, 9).max(dim=2)[0]  # [H,W]
    #             visualize(max_aleatoric, f'visualization/{iIdx}img_{sIdx}scale_maxAle.jpg', size=(oriH, oriW), heatmap=True)
    #             maxAlpha = slvl_alphas.max(dim=1)[0]
    #             maxmaxAlpha = maxAlpha.reshape(H, W, 9).max(dim=2)[0]
    #             visualize(maxmaxAlpha, f'visualization/{iIdx}img_{sIdx}scale_maxmaxAlpha.jpg', size=(oriH, oriW), heatmap=True)
    #             avgmaxAlpha = maxAlpha.reshape(H, W, 9).mean(dim=2)
    #             visualize(avgmaxAlpha, f'visualization/{iIdx}img_{sIdx}scale_avgmaxAlpha.jpg', size=(oriH, oriW), heatmap=True)
    #             selfWmaxEpi = maxmaxAlpha * max_epistemic
    #             visualize(selfWmaxEpi, f'visualization/{iIdx}img_{sIdx}scale_selfWmaxEpi.jpg', size=(oriH, oriW), heatmap=True)
    #             selfWavgEpi = maxmaxAlpha * avg_epistemic
    #             visualize(selfWavgEpi, f'visualization/{iIdx}img_{sIdx}scale_selfWavgEpi.jpg', size=(oriH, oriW), heatmap=True)
    #             selfWavgEpi = (maxmaxAlpha + 1) * avg_epistemic
    #             visualize(selfWavgEpi, f'visualization/{iIdx}img_{sIdx}scale_PlusselfWavgEpi.jpg', size=(oriH, oriW), heatmap=True)


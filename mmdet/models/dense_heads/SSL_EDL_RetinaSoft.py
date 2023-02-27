import pdb
import torch
import torch.nn as nn
from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import HEADS
from .anchor_head import AnchorHead
from mmcv.cnn import ConvModule
from torch.distributions.dirichlet import Dirichlet
from mmdet.utils.functions import *
import pdb

@HEADS.register_module()
class SSLEDLRetinaSoft(AnchorHead):

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
        super(SSLEDLRetinaSoft, self).__init__(
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
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 4.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        return cls_score, bbox_pred

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples, shape=bbox_pred.shape[-2:])
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        if -1 in labels:
            loss_bbox = torch.tensor(1e-9).to(cls_score.device)
        else:
            loss_bbox = self.loss_bbox(
                bbox_pred,
                bbox_targets,
                bbox_weights,
                avg_factor=num_total_samples)

        if self.train_cfg.neg_pos_ratio > 0 :
            pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(as_tuple=False).reshape(-1)
            neg_inds = (labels == self.num_classes).nonzero(as_tuple=False).view(-1)
            num_pos_samples = pos_inds.size(0)
            num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
            if num_neg_samples > neg_inds.size(0):
                num_neg_samples = neg_inds.size(0)
            topk_loss_cls_neg, topk_idces = loss_cls[neg_inds,0].topk(num_neg_samples)
            loss_cls_pos = loss_cls[pos_inds].sum()
            loss_cls_neg = topk_loss_cls_neg.sum()
            loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples
        else:
            loss_cls = loss_cls.sum()

        return loss_cls, loss_bbox


    def _get_bboxes(self,
                    mlvl_cls_scores,
                    mlvl_bbox_preds,
                    mlvl_anchors,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True,
                    **kwargs):

        cfg = self.test_cfg if cfg is None else cfg
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(mlvl_anchors)
        batch_size = mlvl_cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1),
            device=mlvl_cls_scores[0].device,
            dtype=torch.long)

        mlvl_bboxes, mlvl_scores, mlvl_alphas = [], [], []
        for cls_score, bbox_pred, anchors in zip(mlvl_cls_scores,
                                                 mlvl_bbox_preds,
                                                 mlvl_anchors):
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
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if kwargs['isUnc']:
                if kwargs['uPool'] == 'Entropy_ALL':
                    nms_pre = -1
            if nms_pre > 0:
                # Get maximum scores for foreground classes.
                if self.last_activation == 'sigmoid' or self.last_activation == 'relu':
                    max_scores, _ = scores.max(-1)
                elif self.last_activation == 'softmax' or self.last_activation == 'EDL_BG':
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
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
        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            from mmdet.core.export import add_dummy_nms_for_onnx
            # ignore background class
            if self.last_activation == 'softmax':
                num_classes = batch_mlvl_scores.shape[2] - 1
                batch_mlvl_scores = batch_mlvl_scores[..., :num_classes]
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        if self.last_activation == 'relu' or self.last_activation == 'sigmoid':
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = batch_mlvl_scores.new_zeros(batch_size, batch_mlvl_scores.shape[1], 1)
            batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes, batch_mlvl_scores):
                det_bbox, det_label, idces = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                            cfg.score_thr, cfg.nms,
                                                            cfg.max_per_img, return_inds=True)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
            ]
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
            # uncertainties = [i.mean().item() for i in results_list[1]]
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

                # fil_alphas = slvl_alphas[FGIdx]
                # fil_dist = Dirichlet(fil_alphas)
                # samples = fil_dist.sample(torch.tensor([3000]))  # [sample_size, num_bbox, n_way(20)]
                # avg = samples.mean(dim=0)
                # total = (-avg * avg.log()).sum(dim=1)
                # ent = (-samples * samples.log()).sum(dim=-1)
                # aleatoric = ent.mean(dim=0)
                # epistemic = total - aleatoric
                # classes = slvl_alphas[FGIdx].argmax(dim=1).unique()
                # for cls in classes:
                #     clsIdx = slvl_alphas[FGIdx].argmax(dim=1) == cls
                #     clsEpi = epistemic[clsIdx].mean()
                #     clsAle = aleatoric[clsIdx].mean()
                #     print(f'img{iIdx} scale{sIdx} class:{cls}, clsEpi:{clsEpi}, clsAle:{clsAle}')
                # pass

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




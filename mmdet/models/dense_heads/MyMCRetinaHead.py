import torch.nn as nn
import pdb
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from ..builder import HEADS
from .anchor_head import AnchorHead
from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from mmdet.utils.functions import *

@HEADS.register_module()
class MyMCRetinaHead(AnchorHead):
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
        super(MyMCRetinaHead, self).__init__(
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

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights, bbox_targets,
                    bbox_weights, sIdx, num_total_samples, **kwargs):

        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        loss_noR = self.loss_cls(cls_score, labels, reduction_override='none').sum(dim=-1)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_noR

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None, **kwargs):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)

        if self.last_activation == 'sigmoid' or self.last_activation == 'relu':
            label_channels = self.cls_out_channels
        elif self.last_activation == 'softmax' or self.last_activation == 'EDL_BG':
            label_channels = 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        head_info = ['cls_scores', 'bbox_preds', 'all_anchor_list', 'labels_list', 'label_weights_list',
                     'bbox_targets_list', 'bbox_weights_list', 'num_total_samples']
        head_out = (head_info, cls_scores, bbox_preds, all_anchor_list, labels_list, label_weights_list
                    , bbox_targets_list, bbox_weights_list, num_total_samples)

        losses_cls, losses_bbox, losses_noR = multi_apply(self.loss_single,
                                                          cls_scores, bbox_preds, all_anchor_list, labels_list,
                                                          label_weights_list, bbox_targets_list, bbox_weights_list,
                                                          [0, 1, 2, 3, 4],
                                                          num_total_samples=num_total_samples,
                                                          featmap_sizes=featmap_sizes, **kwargs)

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_noR=losses_noR), head_out
    # def _get_bboxes(self, mlvl_cls_scores, mlvl_bbox_preds, mlvl_anchors, img_shapes, scale_factors,
    #                 cfg, rescale=False, with_nms=True, **kwargs):
    #
    #     cfg = self.test_cfg if cfg is None else cfg
    #     assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(mlvl_anchors)
    #     batch_size = mlvl_cls_scores[0].shape[0]
    #     # convert to tensor to keep tracing
    #     nms_pre_tensor = torch.tensor(cfg.get('nms_pre', -1), device=mlvl_cls_scores[0].device, dtype=torch.long)
    #
    #     mlvl_bboxes, mlvl_scores = [], []
    #     for cls_score, bbox_pred, anchors in zip(mlvl_cls_scores, mlvl_bbox_preds, mlvl_anchors):
    #         assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
    #         cls_score = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels)
    #         if self.last_activation == 'sigmoid':
    #             scores = cls_score.sigmoid()
    #         elif self.last_activation == 'relu' or self.last_activation == 'EDL_BG':
    #             alphas = cls_score.relu() + 1
    #             S = alphas.sum(dim=2, keepdim=True) + 1e-20
    #             Smax, _ = S.max(dim=1, keepdim=True)
    #             gamma = 1
    #             scores = alphas / ((1-gamma)*Smax + gamma*S)
    #         elif self.last_activation == 'softmax':
    #             scores = cls_score.softmax(-1)
    #
    #         bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
    #         anchors = anchors.expand_as(bbox_pred)
    #         # Always keep topk op for dynamic input in onnx
    #         from mmdet.core.export import get_k_for_topk
    #         nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
    #         if nms_pre > 0:
    #             # Get maximum scores for foreground classes.
    #             if self.last_activation == 'sigmoid' or self.last_activation == 'relu':
    #                 max_scores, _ = scores.max(-1)
    #             elif self.last_activation == 'softmax' or self.last_activation == 'EDL_BG':
    #                 max_scores, _ = scores[..., :-1].max(-1)
    #
    #             _, topk_inds = max_scores.topk(nms_pre)
    #             batch_inds = torch.arange(batch_size).view(-1, 1).expand_as(topk_inds)
    #             anchors = anchors[batch_inds, topk_inds, :]
    #             bbox_pred = bbox_pred[batch_inds, topk_inds, :]
    #             scores = scores[batch_inds, topk_inds, :]
    #
    #         bboxes = self.bbox_coder.decode(anchors, bbox_pred, max_shape=img_shapes)
    #         mlvl_bboxes.append(bboxes)
    #         mlvl_scores.append(scores)
    #
    #     batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
    #     if rescale:
    #         batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(scale_factors).unsqueeze(1)
    #     batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
    #
    #     # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
    #     if torch.onnx.is_in_onnx_export() and with_nms:
    #         from mmdet.core.export import add_dummy_nms_for_onnx
    #         # ignore background class
    #         if self.last_activation == 'softmax':
    #             num_classes = batch_mlvl_scores.shape[2] - 1
    #             batch_mlvl_scores = batch_mlvl_scores[..., :num_classes]
    #         max_output_boxes_per_class = cfg.nms.get(
    #             'max_output_boxes_per_class', 200)
    #         iou_threshold = cfg.nms.get('iou_threshold', 0.5)
    #         score_threshold = cfg.score_thr
    #         nms_pre = cfg.get('deploy_nms_pre', -1)
    #         return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores, max_output_boxes_per_class,
    #                                       iou_threshold, score_threshold, nms_pre, cfg.max_per_img)
    #     if self.last_activation == 'relu' or self.last_activation == 'sigmoid':
    #         padding = batch_mlvl_scores.new_zeros(batch_size, batch_mlvl_scores.shape[1], 1)
    #         batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)
    #
    #
    #     if with_nms:
    #         det_results = []
    #         for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes, batch_mlvl_scores):
    #             det_bbox, det_label, idces = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms,
    #                                                         cfg.max_per_img, return_inds=True)
    #             det_results.append(tuple([det_bbox, det_label]))
    #     else:
    #         det_results = [tuple(mlvl_bs) for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)]
    #
    #     return det_results
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import force_fp32

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from ..builder import HEADS, build_loss
from ..losses import smooth_l1_loss
from .SSD_anchor_head import SSDAnchorHead
from mmdet.utils.functions import *

@HEADS.register_module()
class SSD_L_Head(SSDAnchorHead):
    """SSD head used in https://arxiv.org/abs/1512.02325.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 0.
        feat_channels (int): Number of hidden channels when stacked_convs
            > 0. Default: 256.
        use_depthwise (bool): Whether to use DepthwiseSeparableConv.
            Default: False.
        conv_cfg (dict): Dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: None.
        act_cfg (dict): Dictionary to construct and config activation layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes=80,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 stacked_convs=0,
                 feat_channels=256,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 anchor_generator=dict(type='SSDAnchorGenerator', scale_major=False, input_size=300, strides=[8, 16, 32, 64, 100, 300],
                     ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]), basesize_ratio_range=(0.1, 0.9)),
                 bbox_coder=dict(type='DeltaXYWHBBoxCoder', clip_border=True, target_means=[.0, .0, .0, .0], target_stds=[1.0, 1.0, 1.0, 1.0]),
                 reg_decoded_bbox=False,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),  # last_activation='sigmoid'
                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform', bias=0)):
        super(SSDAnchorHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.use_depthwise = use_depthwise
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.cls_out_channels = num_classes # add background class
        self.anchor_generator = build_anchor_generator(anchor_generator)
        self.num_anchors = self.anchor_generator.num_base_anchors

        self._init_layers()

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # set sampling=False for archor_target
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs, self.reg_convs, self.L_convs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        conv = DepthwiseSeparableConvModule if self.use_depthwise else ConvModule

        for channel, num_anchors in zip(self.in_channels, self.num_anchors):
            cls_layers, reg_layers, L_layers = [], [], []
            in_channel = channel
            # build stacked conv tower, not used in default ssd
            for i in range(self.stacked_convs):
                cls_layers.append(conv(in_channel, self.feat_channels, 3, padding=1, conv_cfg=self.conv_cfg,
                                       norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
                reg_layers.append(conv(in_channel, self.feat_channels, 3, padding=1, conv_cfg=self.conv_cfg,
                                       norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
                L_layers.append(conv(in_channel, self.feat_channels, 3, padding=1, conv_cfg=self.conv_cfg,
                                       norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
                in_channel = self.feat_channels
            if self.use_depthwise:
                cls_layers.append(ConvModule(in_channel, in_channel, 3, padding=1, groups=in_channel, conv_cfg=self.conv_cfg,
                                             norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
                reg_layers.append(ConvModule(in_channel, in_channel, 3, padding=1, groups=in_channel, conv_cfg=self.conv_cfg,
                                             norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
                L_layers.append(ConvModule(in_channel, in_channel, 3, padding=1, groups=in_channel, conv_cfg=self.conv_cfg,
                                             norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))            # SSD-Lite head
            cls_layers.append(nn.Conv2d(in_channel, num_anchors * self.cls_out_channels, kernel_size=1 if self.use_depthwise else 3,
                                        padding=0 if self.use_depthwise else 1))
            reg_layers.append(nn.Conv2d(in_channel, num_anchors * 4, kernel_size=1 if self.use_depthwise else 3,
                                        padding=0 if self.use_depthwise else 1))
            L_layers.append(nn.Conv2d(in_channel, num_anchors, kernel_size=1 if self.use_depthwise else 3,
                                        padding=0 if self.use_depthwise else 1))
            self.cls_convs.append(nn.Sequential(*cls_layers))
            self.reg_convs.append(nn.Sequential(*reg_layers))
            self.L_convs.append(nn.Sequential(*L_layers))

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=None, proposal_cfg=None, **kwargs):
        outs = self.forward(x, **kwargs)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
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
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        cls_scores, bbox_preds = [], []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs, self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds

    def forward_L(self, feats, head_out, **kwargs):
        L_scores = []
        for feat, L_conv in zip(feats, self.L_convs):
            L_score = L_conv(feat)
            L_scores.append(F.relu(L_score))
        return L_scores

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, sIdx, num_total_samples, **kwargs):

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_noR = self.loss_cls(cls_score, labels, reduction_override='none').sum(dim=-1)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples).sum()
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples)
        return loss_cls, loss_bbox, loss_noR

    @force_fp32(apply_to=('L_score'))
    def loss_single_L(self, L_score, loss, label_weights, bbox_weights, **kwargs):
        weights = bbox_weights[..., 0]  # Or it can be just label_weights. (It was..! for 72.10)
        weights = weights.reshape(-1)
        L_score2 = L_score.permute(0, 2, 3, 1).reshape(-1) + 1e-9
        loss_L = torch.abs(L_score2 - loss)
        loss_L = (loss_L * weights).pow(2).mean() * 5
        def visualize_lossL_info():
            B, C, H, W = L_score.shape
            b, _, h, w = kwargs['_data']['img'].data[0].shape
            for i in range(B):
                visualize(kwargs['_data']['img'].data[0][i], f'visualization/img_{i}.jpg')
                visualize(L_score.permute(0, 2, 3, 1)[i].sum(dim=-1), f'visualization/L_{i}.jpg', size=(h, w), heatmap=True)
                visualize(label_weights.reshape(B, H, W, C)[i].sum(dim=-1), f'visualization/weight_{i}.jpg', size=(h, w), heatmap=True)
                visualize(loss.reshape(B, H, W, C)[i].sum(dim=-1), f'visualization/prevLoss_{i}.jpg', size=(h, w), heatmap=True)
        # visualize_lossL_info()
        return loss_L, 0

    # @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    # def loss_single(self, cls_score, bbox_pred, anchor, labels, label_weights, bbox_targets, bbox_weights, num_total_samples, **kwargs):
    #     loss_cls_all = F.cross_entropy(cls_score, labels, reduction='none') * label_weights
    #     # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    #     pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(as_tuple=False).reshape(-1)
    #     neg_inds = (labels == self.num_classes).nonzero(as_tuple=False).view(-1)
    #
    #     num_pos_samples = pos_inds.size(0)
    #     num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
    #     if num_neg_samples > neg_inds.size(0):
    #         num_neg_samples = neg_inds.size(0)
    #     topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
    #     loss_cls_pos = loss_cls_all[pos_inds].sum()
    #     loss_cls_neg = topk_loss_cls_neg.sum()
    #     loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples
    #
    #     if self.reg_decoded_bbox:
    #         bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)
    #
    #     loss_bbox = smooth_l1_loss(bbox_pred, bbox_targets, bbox_weights,
    #                                beta=self.train_cfg.smoothl1_beta, avg_factor=num_total_samples)
    #     return loss_cls[None], loss_bbox

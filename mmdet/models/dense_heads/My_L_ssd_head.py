import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import force_fp32
from torch.distributions.dirichlet import Dirichlet
from mmdet.core import (build_anchor_generator, build_assigner, multiclass_nms,
                        build_bbox_coder, build_sampler, multi_apply)
from ..builder import HEADS
from ..losses import smooth_l1_loss
from .My_anchor_head import MyAnchorHead
from mmdet.utils.functions import *
from mmdet.core.export import get_k_for_topk
from mmdet.core.export import add_dummy_nms_for_onnx

global posAccList
global negAccList
posAccList, negAccList = [] , []
ignoreBG = False
print(f'=========== ignoreBG is {ignoreBG} ===========')

# TODO: add loss evaluator for SSD
@HEADS.register_module()
class MyLSSDHead(MyAnchorHead):
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
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform', bias=0)):
        super(MyAnchorHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.use_depthwise = use_depthwise
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.last_activation = 'softmax'
        self.cls_out_channels = num_classes + 1  # add background class
        self.anchor_generator = build_anchor_generator(anchor_generator)
        self.num_anchors = self.anchor_generator.num_base_anchors

        self._init_layers()

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
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
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.L_convs = nn.ModuleList()
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
            # SSD-Lite head
            if self.use_depthwise:
                cls_layers.append(ConvModule(in_channel, in_channel, 3, padding=1, groups=in_channel, conv_cfg=self.conv_cfg,
                                             norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
                reg_layers.append(ConvModule(in_channel, in_channel, 3, padding=1, groups=in_channel, conv_cfg=self.conv_cfg,
                                             norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
                L_layers.append(ConvModule(in_channel, in_channel, 3, padding=1, groups=in_channel, conv_cfg=self.conv_cfg,
                                           norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
            cls_layers.append(nn.Conv2d(in_channel, num_anchors * self.cls_out_channels,
                                        kernel_size=1 if self.use_depthwise else 3, padding=0 if self.use_depthwise else 1))
            reg_layers.append(nn.Conv2d(in_channel, num_anchors * 4, kernel_size=1 if self.use_depthwise else 3,
                                        padding=0 if self.use_depthwise else 1))
            L_layers.append(nn.Conv2d(in_channel, num_anchors, kernel_size=1 if self.use_depthwise else 3,
                                     padding=0 if self.use_depthwise else 1))
            self.cls_convs.append(nn.Sequential(*cls_layers))
            self.reg_convs.append(nn.Sequential(*reg_layers))
            self.L_convs.append(nn.Sequential(*L_layers))

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=None, proposal_cfg=None, **kwargs):

        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses, head_out = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, **kwargs) # next? make loss_noR
        if proposal_cfg is None:
            return losses, head_out
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, head_out, proposal_list

    def forward_train_L(self, prev_loss, head_out, x, **kwargs):
        L_scores = self.forward_L(x, head_out, **kwargs)
        losses = self.loss_L(L_scores, head_out, prev_loss, **kwargs)
        def vis_img_lscore_loss():
            img = kwargs['_data']['img'].data[0][0]
            _, H, W = img.shape
            visualize(img, f'img.jpg')
            for sl_loss, sl_l, sl_feat, sIdx in zip(prev_loss, L_scores, x, range(5)):
                _, _, h, w = sl_feat.shape
                loss0 = sl_loss.reshape(2, h, w, 9)[0].sum(dim=-1)
                lambda0 = sl_l[0].permute([1, 2, 0]).sum(dim=-1)
                visualize(loss0, f'loss{sIdx}.jpg', color=False, size=(H, W))
                visualize(lambda0, f'lambda{sIdx}.jpg', color=False, size=(H, W))

        return losses

    def forward(self, feats, **kwargs):
        cls_scores, bbox_preds = [], []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs, self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds

    def forward_L(self, feats, head_out, **kwargs):
        L_scores = []
        for feat, L_conv in zip(feats, self.L_convs):
            L_scores.append(F.relu(L_conv(feat)))
        return L_scores

    def loss_single(self, cls_score, bbox_pred, anchor, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        loss_cls_all = F.cross_entropy(cls_score, labels, reduction='none') * label_weights
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(as_tuple=False).reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero(as_tuple=False).view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, topIdx = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples
        # if len(pos_inds) != 0:
        #     posAccList.append((cls_score[pos_inds].argmax(dim=1) == labels[pos_inds]).float().mean())
        #     negAccList.append((cls_score[topIdx].argmax(dim=1) == labels[topIdx]).float().mean())
        #     if len(posAccList) == 50:
        #         print(f'posAcc : {torch.tensor(posAccList).mean()}')
        #         print(f'negAcc : {torch.tensor(negAccList).mean()}')
        #         posAccList.clear()
        #         negAccList.clear()

        if self.reg_decoded_bbox:
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox, loss_cls_all

    def loss_single_L(self, L_score, prev_loss, weight, **kwargs):
        loss_L = torch.abs(L_score - prev_loss)
        if 'mineW' in kwargs and kwargs['mineW']:
            loss_L = (loss_L * weight).pow(2).mean()
        else:
            loss_L = loss_L.pow(2).mean()

        return loss_L*2, 0

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None, **kwargs):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            # unmap_outputs=False,
            unmap_outputs=True,
        )
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))

        # check NaN and Inf
        assert torch.isfinite(all_cls_scores).all().item(), 'classification scores become infinite or NaN!'
        assert torch.isfinite(all_bbox_preds).all().item(), 'bbox predications become infinite or NaN!'


        head_info = ['cls_scores', 'bbox_preds', 'all_anchor_list', 'labels_list', 'label_weights_list',
                     'bbox_targets_list', 'bbox_weights_list', 'num_total_samples']
        head_out = (head_info, cls_scores, bbox_preds, None, labels_list, label_weights_list
                    , bbox_targets_list, bbox_weights_list, None)

        losses_cls, losses_bbox, losses_noR = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_noR=losses_noR), head_out

    def loss_L(self, L_scores, head_out, prev_loss, **kwargs):
        bbox_weights_list = head_out[7]
        all_weights = torch.cat([w[..., 0] for w in bbox_weights_list], 1)
        all_L_scores = torch.cat([l.permute(0,2,3,1).reshape(8,-1) for l in L_scores], 1) + 1e-9
        all_prev_loss = torch.stack(prev_loss)
        loss_L, _ = multi_apply(
            self.loss_single_L,
            all_L_scores,
            all_prev_loss,
            all_weights,
            )
        return dict(loss_L=loss_L)

    @force_fp32(apply_to=('mlvl_cls_scores', 'mlvl_bbox_preds', 'mlvl_anchors'))
    def _get_bboxes(self, mlvl_cls_scores, mlvl_bbox_preds, mlvl_anchors, img_shapes, scale_factors,
                    cfg, rescale=False, with_nms=True, **kwargs):

        cfg = self.test_cfg if cfg is None else cfg
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(mlvl_anchors)
        batch_size = mlvl_cls_scores[0].shape[0]
        nms_pre_tensor = torch.tensor(cfg.get('nms_pre', -1), device=mlvl_cls_scores[0].device, dtype=torch.long)
        L_scores = kwargs['L_scores'] if 'L_scores' in kwargs else mlvl_cls_scores
        mlvl_bboxes, mlvl_scores, mlvl_Ls = [], [], []
        for cls_score, bbox_pred, anchors, l_scores in zip(mlvl_cls_scores, mlvl_bbox_preds, mlvl_anchors, L_scores):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels)
            l_scores = l_scores.permute(0,2,3,1).reshape(batch_size, -1)
            if self.last_activation == 'softmax':
                scores = cls_score.softmax(-1)
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
                elif self.last_activation == 'softmax':
                    max_scores, _ = scores[..., :-1].max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(-1, 1).expand_as(topk_inds)
                anchors = anchors[batch_inds, topk_inds, :]
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]
                l_scores = l_scores[batch_inds, topk_inds]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
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
                objUnc = self.ComputeObjUnc(mlvl_cls_scores, pos_bboxes, bak_mlvl_scores, mlvl_Ls, None)
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

    def ComputeObjUnc(self, mlvl_cls_scores, pos_bboxes, mlvl_scores, mlvl_Ls, mlvl_idces):
        S = len(mlvl_cls_scores)
        B, _, oriH, oriW = mlvl_cls_scores[0].shape
        lenObj = [i.size(1) for i in pos_bboxes]
        output = [[[{} for _ in range(S)] for _ in range(lenObj[b])] for b in range(B)]
        for sIdx, slvl_scores in enumerate(mlvl_cls_scores):
            for iIdx, simg_scores in enumerate(slvl_scores):
                simg_scores = simg_scores.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
                slvl_alphas = simg_scores.softmax(dim=1)  # [ N, 20]
                slvl_maxconf = slvl_alphas[:,:-1].max(dim=1)[0]
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
                pos_alphas = pos_scores
                pos_alphas = pos_alphas * pos_l_scores[:, None]
                if ignoreBG: pos_dist = Dirichlet(pos_alphas[:, :-1])
                else: pos_dist = Dirichlet(pos_alphas)
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
                slvl_maxconf = slvl_alphas[:,:-1].max(dim=1)[0]
                FGIdx, BGIdx = (slvl_maxconf > 0.3), (slvl_maxconf <= 0.3)
                if not True in FGIdx:
                    continue
                l_scores = L_scores[sIdx][iIdx].permute(1, 2, 0).reshape(-1, 1)
                eps = 1e-7
                l_scores = l_scores.mean() / (l_scores + eps) * 25
                slvl_alphas = slvl_alphas * l_scores
                fil_alphas = slvl_alphas[FGIdx]
                if ignoreBG: fil_dist = Dirichlet(fil_alphas[:, :-1])
                else: fil_dist = Dirichlet(fil_alphas)
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

    def simple_test(self, feats, img_metas, rescale=False, **kwargs):
        # input : self.bbox_head.simple_test(feat, img_metas, rescale=rescale, **kwargs)
        outs = self.forward(feats)
        L_scores = self.forward_L(feats, head_out=None)
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
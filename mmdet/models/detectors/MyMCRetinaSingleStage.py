import warnings
import pdb
import torch

from mmdet.core import bbox2result, bbox2tupleresult
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .MyRetinaBase import MyRetinaBase
from mmdet.utils.functions import *

@DETECTORS.register_module()
class MyMCRetinaSingleStageDetector(MyRetinaBase):
    def __init__(self, backbone, neck=None, bbox_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        super(MyMCRetinaSingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs):
        super(MyMCRetinaSingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses, head_out = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, **kwargs)
        feat_out = [i.detach() for i in x]
        return losses, head_out, feat_out

    def forward_train_L(self, loss, head_out, feat_out, **kwargs,):
        losses = self.bbox_head.forward_train_L(loss, head_out, feat_out, **kwargs)
        return losses

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        import sys
        sys.setrecursionlimit(10**7)
        if 'justOut' in kwargs and kwargs['justOut']:
            feat = self.extract_feat(img)
            _results_list = self.bbox_head.simple_test(feat, img_metas, rescale=rescale, **kwargs)
            return _results_list[0]
        outList = []
        for _ in range(3): # to be 25
            feat = self.extract_feat(img)
            out = self.bbox_head.forward(feat)
            outList.append(out) # tuple of length 2, each containing cls_out & reg_out
        clsOuts, regOuts = list(zip(*outList))
        clsScaleOuts = list(zip(*clsOuts))
        regScaleOuts = list(zip(*regOuts))
        avgClsOut = [torch.cat(clsOuts).mean(dim=0, keepdim=True) for clsOuts in clsScaleOuts]
        avgRegOut = [torch.cat(regOuts).mean(dim=0, keepdim=True) for regOuts in regScaleOuts]
        avgOut = (avgClsOut, avgRegOut)

        if kwargs['isEval']:
            # _results_list = self.bbox_head.simple_test(feat, img_metas, rescale=rescale, **kwargs)
            _results_list = self.bbox_head.get_bboxes(*avgOut, img_metas, rescale=rescale, **kwargs)
            if kwargs['isUnc']:
                results_list, entropy_list = _results_list[0], _results_list[1]
                bbox_results = [
                    bbox2tupleresult(det_bboxes, det_labels, entropy_list, self.bbox_head.num_classes)
                    for det_bboxes, det_labels in results_list
                ]
            elif not kwargs['isUnc']:
                results_list = _results_list
                bbox_results = [
                    bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                    for det_bboxes, det_labels in results_list
                ]
            return bbox_results
        else:
            results_list, *uncertainties = self.bbox_head.simple_test(feat, img_metas, rescale=rescale, **kwargs)
            if self.test_cfg.uncertainty_pool == 'Entropy_NoNMS':
                return (results_list, *uncertainties)
            elif self.test_cfg.uncertainty_pool == 'Entropy_ALL':
                return (results_list, *uncertainties)
            elif self.test_cfg.uncertainty_pool == 'Entropy_NMS':
                return (results_list, *uncertainties)
            bbox_results = [bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                            for det_bboxes, det_labels in results_list]
            return bbox_results, uncertainties

    def MC_simple_test(self, img, img_metas, rescale=False, **kwargs):
        feat = self.extract_feat(img)
        out = self.bbox_head.forward(feat)
        _results_list = self.bbox_head.simple_test(feat, img_metas, rescale=rescale, **kwargs)
        if 'justOut' in kwargs and kwargs['justOut']:
            return _results_list[0]
        results_list = _results_list
        bbox_results = [bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                        for det_bboxes, det_labels in results_list]
        return bbox_results


    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas)

        return det_bboxes, det_labels

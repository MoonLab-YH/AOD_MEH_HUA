from ..builder import DETECTORS
from .SelfSSL_single_stage import SelfSSL_SingleStageDetector


@DETECTORS.register_module()
class SelfSSL_RetinaNet(SelfSSL_SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SelfSSL_RetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained, init_cfg)

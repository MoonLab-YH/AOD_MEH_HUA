from ..builder import DETECTORS
from .MyMCRetinaSingleStage import MyMCRetinaSingleStageDetector


@DETECTORS.register_module()
class MyMCRetinaNet(MyMCRetinaSingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(MyMCRetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained, init_cfg)

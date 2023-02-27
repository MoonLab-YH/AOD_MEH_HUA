from ..builder import DETECTORS
from .SSL_L_single_stage import SSL_L_SingleStageDetector


@DETECTORS.register_module()
class SSL_L_RetinaNet(SSL_L_SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SSL_L_RetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained, init_cfg)

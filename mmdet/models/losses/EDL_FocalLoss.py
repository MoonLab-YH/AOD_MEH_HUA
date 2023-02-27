import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from ..builder import LOSSES
from .utils import weight_reduce_loss

def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):

    loss = _sigmoid_focal_loss(pred.contiguous(), target, gamma, alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class EDL_FocalLoss(nn.Module):

    def __init__(self,
                 num_classes,
                 annealing_step,
                 last_activation='sigmoid',
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 ):
        super(EDL_FocalLoss, self).__init__()
        self.last_activation = last_activation
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        evidence = F.relu(pred)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / S
        logits = (prob / (1-prob)).log()

        loss_cls = self.loss_weight * sigmoid_focal_loss(
            logits, target, weight, gamma=self.gamma, alpha=self.alpha,
            reduction=reduction, avg_factor=avg_factor)

        return loss_cls

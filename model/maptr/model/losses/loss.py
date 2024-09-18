import functools

import torch
from mmcv.utils import jit
from mmdet.models import weighted_loss
from torch import nn


class SimpleLoss(nn.Module):
    def __init__(self, pos_weight, loss_weight):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
        self.loss_weight = loss_weight

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss * self.loss_weight


class PtsL1Loss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PtsL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * pts_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


class PtsDirCosLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PtsDirCosLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_weight * pts_dir_cos_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_dir


class PtsL1Cost:
    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        num_pts = gt_bboxes.shape[-2]
        num_coords = gt_bboxes.shape[-1]
        bbox_pred = bbox_pred.reshape(bbox_pred.size(0), -1).contiguous()
        gt_bboxes = gt_bboxes.reshape(-1, num_pts * num_coords).contiguous()
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


class FocalCost:
    def __init__(self, weight=1., alpha=0.25, gamma=2, eps=1e-12):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
                1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
                1 - cls_pred).pow(self.gamma)
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight


@jit(derivate=True, coderize=True)
def custom_weight_dir_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        raise ValueError('avg_factor should not be none for OrderedPtsL1Loss')
    else:
        if reduction == 'mean':
            loss = loss.sum()
            loss = loss / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def custom_weighted_dir_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = custom_weight_dir_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


@jit(derivate=True, coderize=True)
@custom_weighted_dir_loss
def pts_dir_cos_loss(pred, target):
    """ Dir cosine similiarity loss
    pred (torch.Tensor): shape [num_samples, num_dir, num_coords]
    target (torch.Tensor): shape [num_samples, num_dir, num_coords]

    """
    if target.numel() == 0:
        return pred.sum() * 0
    num_samples, num_dir, num_coords = pred.shape
    loss_func = torch.nn.CosineEmbeddingLoss(reduction='none')
    tgt_param = target.new_ones((num_samples, num_dir))
    tgt_param = tgt_param.flatten(0)
    loss = loss_func(pred.flatten(0, 1), target.flatten(0, 1), tgt_param)
    loss = loss.view(num_samples, num_dir)
    return loss


@jit(derivate=True, coderize=True)
@weighted_loss
def pts_l1_loss(pred, target):
    if target.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss

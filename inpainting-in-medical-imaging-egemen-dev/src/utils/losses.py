from typing import List

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class TargetArgmaxLossWrapper(_Loss):
    def __init__(self, loss, keep_dims=False):
        super().__init__()
        self.loss = loss
        self.keep_dims = keep_dims

    def forward(self, predict: torch.Tensor, target: torch.Tensor):
        return self.loss(predict, target.argmax(1, self.keep_dims))


class WeightedLoss(_Loss):
    r"""
    TODO documentation
    """

    def __init__(self, loss, weight: float = 1.0):
        super().__init__()
        self.loss = loss
        self.w = weight

    def forward(self, *x):
        return self.loss(*x) * self.w


class JointLoss(_Loss):
    r"""
    TODO documentation
    """

    def __init__(self, losses: List[_Loss], weights: List[float]):
        super().__init__()
        assert len(losses) == len(weights)
        self.losses = [WeightedLoss(x, weights[i]) for i, x in enumerate(losses)]

    def forward(self, *x):
        return sum([loss(*x) for loss in self.losses])


class BinaryDiceLoss(nn.Module):
    r"""
    TODO documentation
    """

    def __init__(self, smooth=1e-6, p=2, reduction="mean"):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert (
            predict.shape[0] == target.shape[0]
        ), f"predict {predict.shape[0]} & target {target.shape[0]} batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = (2 * torch.sum(torch.mul(predict, target), dim=1)) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise Exception("Unexpected reduction {}".format(self.reduction))


class DiceLoss(nn.Module):
    r"""
    TODO documentation
    """

    def __init__(
        self, weights=None, ignore_index=None, apply_softmax: bool = True, **kwargs
    ):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weights
        self.apply_softmax = apply_softmax
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert (
            predict.shape == target.shape
        ), f"predict {predict.shape} & target {target.shape} sizes don't match"
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        if self.apply_softmax:
            predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert (
                        self.weight.shape[0] == target.shape[1]
                    ), "Expect weight shape [{}], get[{}]".format(
                        target.shape[1], self.weight.shape[0]
                    )
                    dice_loss *= self.weights[i]
                total_loss += dice_loss
        return total_loss / target.shape[1]

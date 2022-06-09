from typing import Literal, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
from sklearn.metrics import confusion_matrix


def make_one_hot(input, num_classes, device="cpu"):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).to(device)
    result = result.scatter_(1, input, 1)
    return result


def calc_confusion_matrix(
    target: torch.Tensor,
    pred: torch.Tensor,
    n_classes: int,
    normalize: bool = False,
    reduction: Optional[Union[Literal["mean"], Literal["sum"]]] = None,
) -> Union[int, float, npt.ArrayLike]:
    r"""
    TODO documentation
    """
    assert target.shape == pred.shape

    t = target.contiguous().view(target.shape[0], -1)
    p = pred.contiguous().view(pred.shape[0], -1)

    out = np.empty((target.shape[0], n_classes, n_classes))
    for i in range(target.shape[0]):
        cm = confusion_matrix(t[i], p[i], labels=range(n_classes), normalize=None)
        if normalize:
            cm = np.true_divide(cm, cm.sum(axis=1)[:, np.newaxis])
            cm[np.isnan(cm)] = 0
        out[i] = cm

    if reduction == "mean":
        return np.mean(out, axis=0)
    elif reduction == "sum":
        return np.sum(out, axis=0)
    else:
        return out


def calc_precision(
    confusion_matrix: npt.ArrayLike,
    reduction: Optional[Union[Literal["mean"], Literal["sum"]]] = None,
):
    r"""
    TODO documentation
    TP / (TP + FP)
    """
    np.seterr(invalid="ignore")
    cm = confusion_matrix.copy()
    if confusion_matrix.ndim <= 2:
        cm = np.expand_dims(cm, axis=0)
    out = np.empty((cm.shape[0], cm.shape[1]))
    for i in range(out.shape[0]):
        out[i] = np.true_divide(np.diag(cm[i]), np.sum(cm[i], axis=0))
        out[i][np.isnan(out[i])] = 1
    if reduction == "mean":
        return np.mean(out, axis=0)
    elif reduction == "sum":
        return np.sum(out, axis=0)
    else:
        return out


def calc_recall(
    confusion_matrix,
    reduction: Optional[Union[Literal["mean"], Literal["sum"]]] = None,
):
    r"""
    TODO documentation
    TP / (FN + TP)
    """
    np.seterr(invalid="ignore")
    cm = confusion_matrix.copy()
    if confusion_matrix.ndim <= 2:
        cm = np.expand_dims(cm, axis=0)
    out = np.empty((cm.shape[0], cm.shape[1]))
    for i in range(out.shape[0]):
        out[i] = np.true_divide(np.diag(cm[i]), np.sum(cm[i], axis=1))
        out[i][np.isnan(out[i])] = 1
    if reduction == "mean":
        return np.mean(out, axis=0)
    elif reduction == "sum":
        return np.sum(out, axis=0)
    else:
        return out


# TODO Revise Losses


class ExponentialLogarithmicLoss(nn.Module):
    def __init__(self, device):
        super(ExponentialLogarithmicLoss, self).__init__()
        # self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        # self.w_dice = torch.tensor([1.0]).requires_grad_().to(device)
        # self.w_ce = torch.tensor([1.0]).requires_grad_().to(device)
        self.w_dice = torch.nn.Parameter(torch.ones(2)).requires_grad_().to(device)
        self.w_ce = torch.nn.Parameter(torch.ones(2)).requires_grad_().to(device)

    def forward(self, predict, target):
        return self.w_dice * dice_loss(
            torch.softmax(predict, 1), target, True
        ) + self.w_ce * self.ce_loss(predict, target)


class BinaryDiceLoss(nn.Module):
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
    def __init__(
        self, weight=None, ignore_index=None, apply_softmax: bool = True, **kwargs
    ):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.apply_softmax = apply_softmax
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert (
            predict.shape == target.shape
        ), f"predict {predict.shape} & target {target.shape} batch size don't match"
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


def dice_coeff(
    input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6
):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(
            f"Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})"
        )

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(
    input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6
):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(
            input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon
        )

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


# TOOO remove this
MSELoss = torch.nn.MSELoss()


def realTargetLoss(x):
    r"""
    Transport this to corresponding experiment file
    """
    device = x.device
    target = Tensor(x.shape[0], 1).fill_(1.0).to(device)
    return MSELoss(x, target)


def fakeTargetLoss(x):
    r"""
    Transport this to corresponding experiment file
    """
    device = x.device
    target = Tensor(x.shape[0], 1).fill_(0.0).to(device)
    return MSELoss(x, target)

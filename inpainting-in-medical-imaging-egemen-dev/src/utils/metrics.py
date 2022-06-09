from typing import List, Literal, Optional, Union

import torch
import numpy as np
import numpy.typing as npt
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

from src.utils.losses import BinaryDiceLoss


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


BDICE_LOSS = BinaryDiceLoss(reduction=None)


def calc_dice_score(
    p: torch.Tensor,
    t: torch.Tensor,
    reduction: Optional[Union[Literal["mean"], Literal["sum"]]] = None,
):
    x = F.softmax(p, dim=1).argmax(1, True)
    out = torch.zeros((x.shape[0], t.shape[1]))
    for i in range(t.shape[1]):
        predict = torch.where(x == i, 1.0, 0.0)
        predict = predict.contiguous().view(p.shape[0], -1)
        target = t[:, i].contiguous().view(t.shape[0], -1)
        num = (2 * torch.sum(torch.mul(predict, target), dim=1)) + 1e-6
        den = torch.sum(predict.pow(2) + target.pow(2), dim=1) + 1e-6
        out[:, i] = num / den
    if reduction == "mean":
        return torch.mean(out, axis=0)
    elif reduction == "sum":
        return torch.sum(out, axis=0)
    else:
        return out

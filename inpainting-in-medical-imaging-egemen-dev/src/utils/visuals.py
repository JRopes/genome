from typing import List

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import numpy.typing as npt
import torch
from torchvision.utils import make_grid

from src.utils.tumor import binarize, quantize


def create_grid_grade(
    predict: torch.Tensor, target: torch.Tensor, nrow: int, pad_value=1
) -> torch.Tensor:
    r"""
    TODO documentation
    """

    t = (
        torch.cat(
            (
                quantize(predict.argmax(1, True))[0:nrow, :, :, :],
                quantize(target)[0:nrow, :, :, :],
            ),
            0,
        )
        if predict.shape[1] != 1
        else torch.cat((predict[0:nrow, :, :, :], target[0:nrow, :, :, :]), 0)
    )

    grid = make_grid(t, nrow=nrow, pad_value=pad_value)
    return grid


def create_grid_shape(
    predict: torch.Tensor,
    target: torch.Tensor,
    nrow: int,
    pad_value=1,
    threshold: float = 0.5,
) -> torch.Tensor:
    r"""
    TODO documentation
    """
    grid = make_grid(
        torch.cat(
            (
                binarize(predict, threshold)[0:nrow, :, :, :],
                binarize(target, threshold)[0:nrow, :, :, :],
            ),
            0,
        ),
        nrow=nrow,
        pad_value=pad_value,
    )
    return grid


def create_grid_inpainting(
    predict: torch.Tensor, target: torch.Tensor, nrow: int, pad_value=1
) -> torch.Tensor:
    r"""
    TODO documentation
    """
    t = torch.cat(
        (
            torch.cat(
                (predict[0:nrow, 0, :, :], target[0:nrow, 0, :, :]), dim=2
            ).unsqueeze(1),
            torch.cat(
                (predict[0:nrow, 1, :, :], target[0:nrow, 1, :, :]), dim=2
            ).unsqueeze(1),
            torch.cat(
                (predict[0:nrow, 2, :, :], target[0:nrow, 2, :, :]), dim=2
            ).unsqueeze(1),
            torch.cat(
                (predict[0:nrow, 3, :, :], target[0:nrow, 3, :, :]), dim=2
            ).unsqueeze(1),
        ),
        0,
    )
    grid = make_grid(t, nrow=nrow, pad_value=pad_value)
    return grid


def create_grid_segmentation(
    sample: torch.Tensor,
    predict: torch.Tensor,
    target: torch.Tensor,
    nrow: int,
    pad_value=1,
) -> torch.Tensor:
    r"""
    TODO documentation FIX THIS
    """
    p = predict.argmax(1, True)
    t = torch.cat(
        (sample[0:nrow, 0, :, :], p[0:nrow, 0, :, :], target[0:nrow, 0, :, :].float(),),
        dim=2,
    ).unsqueeze(1)
    grid = make_grid(t, nrow=nrow, pad_value=pad_value)
    return grid


def create_confusion_matrix_image(
    matrix: npt.ArrayLike, labels: List[str]
) -> plt.figure:
    r"""
    Create a confusion image where it shows both numbers as well as normalized values

    Args:
        matrix (npt.Array): confusion matrix.
        labels (list[str]): list of labels
    """
    m = matrix.copy()
    normalized = m / m.sum(axis=1)[:, np.newaxis]
    data_labels = [[0] * m.shape[1] for _ in range(m.shape[0])]
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            data_labels[i][j] = f"{m[i][j]:.0f}\n{normalized[i][j]:.1%}"

    plt.clf()
    hm = sns.heatmap(
        normalized,
        annot=data_labels,
        fmt="",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        xticklabels=labels,
        yticklabels=labels,
        mask=np.isnan(normalized),
    )
    plt.close()
    return hm.get_figure()

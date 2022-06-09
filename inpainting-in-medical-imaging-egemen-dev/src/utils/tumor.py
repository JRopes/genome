from enum import Enum

import numpy as np
import torch


class TumorMask(Enum):
    """Enum class to differentiate tumor levels
    """

    NET = 3  # Necrotic and Non-enhancing Tumor
    ET = 2  # Enhancing Tumor
    ED = 1  # Edema
    WHOLE = 0  # Background


def binarize(array, treshold: float = 0.0):
    out = np.zeros(array.shape)
    out[np.where(array > treshold)] = 1.0
    return torch.from_numpy(out).float()


def quantize_baseline1(array):
    out = np.zeros(array.shape)
    out[np.where(array >= 0.85)] = 1
    out[np.where(array > 0.63) and np.where(array < 0.85)] = 0.75
    out[np.where(array > 0.36) and np.where(array <= 0.63)] = 0.5
    out[np.where(array <= 0.36)] = 0
    return torch.from_numpy(out).float()


def quantize(array):
    out = np.zeros(array.shape)
    out[np.where(array == 1)] = 0.5
    out[np.where(array == 2)] = 0.75
    out[np.where(array == 3)] = 1.0

    return (
        torch.from_numpy(out).float().to(array.get_device())
        if type(array) is torch.Tensor and array.is_cuda
        else torch.from_numpy(out).float()
    )


def map_grade_to_float(x: torch.Tensor) -> torch.Tensor:
    r"""
    Does mapping of tumor grade mask from int values to float values

    Args:
        x (torch.Tensor): input grade mask tensor
    """
    out = torch.zeros(x.shape, dtype=torch.float).to(x.device)
    out[(x == 1).nonzero(as_tuple=True)] = 0.5
    out[(x == 2).nonzero(as_tuple=True)] = 0.75
    out[(x == 3).nonzero(as_tuple=True)] = 1.0
    return out

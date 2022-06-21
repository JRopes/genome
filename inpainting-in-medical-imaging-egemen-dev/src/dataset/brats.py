import pathlib
import glob
import os
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

from src.utils.tumor import TumorMask


class BraTSDataset(Dataset):
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        mode: str,
        tumor_type: str = "hgg-lgg",
        resize=(256, 256),
        fixed_indices: Optional[List] = None,
        add_noise_to_circle_values: Optional[Tuple[float, float]] = None,
        add_noise_to_circle_radius: Optional[Tuple[float, float]] = None,
    ):
        r"""
        Dataset specialized for BraTS18 & BraTS20

        Args:
            root (Union[str, pathlib.Path]): input directory path.
            mode (str): type of usage, "train", "validation" or "test.
            tumor_type (str, optional): hgg, lgg, hgg-lgg. Defaults to "hgg-lgg".
            resize (tuple): desired 2D image size.
            fixed_indices (list, optional): specific indices to be used for loader.
            add_noise_to_circle_values (tuple, optional): mean and std parameters for gaussian noise on circle values.
            add_noise_to_circle_radius (tuple, optional): mean and std parameters for gaussian noise on circle radiuses.
        """
        #self.tumor_type = tumor_type
        #self.tumor_type = hgg
        self.mode = mode
        self.resize = resize
        self.values_noise_parameters = add_noise_to_circle_values
        self.radiuses_noise_parameters = add_noise_to_circle_radius
        print(fixed_indices)
        if fixed_indices is not None:
            self.t1_dirs = [
                os.path.join(root, x.replace("*", "t1")) for x in fixed_indices
            ]
            self.t1ce_dirs = [
                os.path.join(root, x.replace("*", "t1ce")) for x in fixed_indices
            ]
            self.t2_dirs = [
                os.path.join(root, x.replace("*", "t2")) for x in fixed_indices
            ]
            self.flair_dirs = [
                os.path.join(root, x.replace("*", "flair")) for x in fixed_indices
            ]
            self.mask_dirs = [
                os.path.join(root, x.replace("*", "seg")) for x in fixed_indices
            ]
        else:
            self.tumor_type = "hgg-lgg"
            

            self.t1_dirs = sorted(
                glob.glob(
                    os.path.join(
                        root,
                        "*" if self.tumor_type == "hgg-lgg" else self.tumor_type,
                        "*",
                        "t1",
                    )
                    + "/*.png"
                )
            )

            self.t2_dirs = sorted(
                glob.glob(
                    os.path.join(
                        root,
                        "*" if self.tumor_type == "hgg-lgg" else self.tumor_type,
                        "*",
                        "t2",
                    )
                    + "/*.png"
                )
            )

            self.t1ce_dirs = sorted(
                glob.glob(
                    os.path.join(
                        root,
                        "*" if self.tumor_type == "hgg-lgg" else self.tumor_type,
                        "*",
                        "t1ce",
                    )
                    + "/*.png"
                )
            )

            self.flair_dirs = sorted(
                glob.glob(
                    os.path.join(
                        root,
                        "*" if self.tumor_type == "hgg-lgg" else self.tumor_type,
                        "*",
                        "flair",
                    )
                    + "/*.png"
                )
            )

            self.mask_dirs = sorted(
                glob.glob(
                    os.path.join(
                        root,
                        "*" if self.tumor_type == "hgg-lgg" else self.tumor_type,
                        "*",
                        "seg",
                    )
                    + "/*.png"
                )
            )
        segs = self.mask_dirs
        segs = list(map(lambda x: x.replace('seg', 't1'), segs))

        
        not_in_t1 = [i for i in segs if i not in self.t1_dirs]
        not_in_t1 = list(map(lambda x: x.replace('t1', 'seg'), not_in_t1))
        
        self.mask_dirs = [i for i in self.mask_dirs if i not in not_in_t1]
        assert (
            len(self.t1_dirs)
            == len(self.t1ce_dirs)
            == len(self.t2_dirs)
            == len(self.flair_dirs)
            == len(self.mask_dirs)
        )

        self._size = len(self.mask_dirs)

        self.trans_scan = transforms.Compose(
            [
                transforms.Resize(
                    size=(self.resize[0], self.resize[1]),
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )

        self.trans_mask = transforms.Compose(
            [
                transforms.Resize(
                    size=(self.resize[0], self.resize[1]),
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.PILToTensor(),
            ]
        )

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict:
        t1 = self.trans_scan(Image.open(self.t1_dirs[idx]))
        t1ce = self.trans_scan(Image.open(self.t1ce_dirs[idx]))
        t2 = self.trans_scan(Image.open(self.t2_dirs[idx]))
        flair = self.trans_scan(Image.open(self.flair_dirs[idx]))
        m_grade = self.trans_mask(Image.open(self.mask_dirs[idx])).long()
        m_brain = torch.where(flair > 0, 1.0, 0.0)
        m_shape = torch.where(m_grade > 0, 1.0, 0.0)
        m_net = torch.where(m_grade == TumorMask.NET.value, 1.0, 0.0)
        m_et = torch.where(m_grade == TumorMask.ET.value, 1.0, 0.0)
        m_ed = torch.where(m_grade == TumorMask.ED.value, 1.0, 0.0)
        mask_circles = self.create_circles_mask(
            m_grade, self.values_noise_parameters, self.radiuses_noise_parameters
        )

        return {
            "t1": t1,
            "t1ce": t1ce,
            "t2": t2,
            "flair": flair,
            "mask_brain_shape": m_brain,
            "mask_tumor_grade": m_grade,
            "mask_tumor_shape": m_shape,
            "mask_net": m_net,
            "mask_et": m_et,
            "mask_ed": m_ed,
            "mask_circles": mask_circles,
            "mask_circles_org": self.create_circles_mask(m_grade, None, None), # TODO remove this later
        }

    @staticmethod
    def create_circles_mask(
        array: Union[npt.ArrayLike, torch.Tensor],
        values_gaussian_noise_parameters: Optional[Tuple[float, float]] = None,
        radiuses_gaussian_noise_parameters: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        r"""
        Creates a mask with concentric circles based on given mask's values

        Args:
            array (npt.ArrayLike): input mask.
            values_gaussian_noise_parameters (tuple, optional): gaussian noise parameters (mean, std) for values
            radiuses_gaussian_noise_parameters (tuple, optional): gaussian noise parameters (mean, std) for radiuses
        """
        [c, h, w] = array.shape
        nonzero_idx = np.where(array > 0)
        a_whole = np.array(nonzero_idx).shape[1]

        a_ed = np.array(np.where(array >= 1)).shape[1]
        radius_ed = int((a_ed / 3.14) ** (0.5))
        a_et = np.array(np.where(array >= 2)).shape[1]
        radius_et = int((a_et / 3.14) ** (0.5))
        a_net = np.array(np.where(array >= 3)).shape[1]
        radius_net = int((a_net / 3.14) ** (0.5))

        if radiuses_gaussian_noise_parameters is not None:
            mean, std = radiuses_gaussian_noise_parameters
            radius_ed, radius_et, radius_net = add_gaussian_noise_r(
                radius_ed, radius_et, radius_net, mean, std
            )

        # find center
        xx, yy = np.mgrid[:h, :w]
        x_center = np.sum(nonzero_idx[1]) / (a_whole + 0.001)
        y_center = np.sum(nonzero_idx[2]) / (a_whole + 0.001)

        circle = (xx - x_center) ** 2 + (yy - y_center) ** 2

        out = np.zeros((h, w), dtype=np.float32)
        out[np.where(circle < radius_ed ** 2)] = 0.5
        out[np.where(circle < radius_et ** 2)] = 0.75
        out[np.where(circle < radius_net ** 2)] = 1.0

        out = torch.from_numpy(np.reshape(out, [c, h, w]))
        if values_gaussian_noise_parameters is not None:
            mean, std = values_gaussian_noise_parameters
            noise = add_gaussian_noise_v(
                torch.zeros(out.shape, dtype=torch.float64), mean, std
            )
            noise = torch.where(out > 0.0, noise, 0.0)
            out += noise

        return out


def add_gaussian_noise_v(
    x: torch.Tensor, mean: float = 0.0, std: float = 1.0
) -> torch.Tensor:
    r"""
    TODO documentation
    """
    return x + torch.randn(x.size()) * std + mean


def add_gaussian_noise_r(
    ed: int, et: int, net: int, mean: float = 0.0, std: float = 0.1
) -> int:
    r"""
    TODO documentation
    """
    sigma_ed = 2 ** (0.15 * ed) - 1
    sigma_et = 2 ** (0.15 * et) - 1
    sigma_net = 2 ** (0.15 * net) - 1
    r = np.random.normal(loc=mean, scale=std, size=None)
    return (
        max(ed + r * sigma_ed, 0),
        max(et + r * sigma_et, 0),
        max(net + r * sigma_net, 0),
    )

import os
import pathlib
from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np
import nibabel as nib
import cv2

from src.utils.tumor import TumorMask


def get_project_root() -> Path:
    r"""
    Get root path of the project
    """
    return Path(__file__).parent.parent.parent


def mkdir(path: Union[str, pathlib.Path]) -> None:
    r"""
    Create an empty directory

    Args:
        path (Union[str, pathlib.Path]): target directory path
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class NiiFileExtension(Enum):
    r"""
    Nii File Extension enum for different types of nii files
    """

    STANDARD = ("", "nii.gz")
    T1 = ("t1", "t1.nii.gz")
    T2 = ("t2", "t2.nii.gz")
    T1CE = ("t1ce", "t1ce.nii.gz")
    FLAIR = ("flair", "flair.nii.gz")
    SEG = ("seg", "seg.nii.gz")

    @property
    def name(self) -> str:
        return self.value[0]

    @property
    def extension(self) -> str:
        return self.value[1]


def extract_images_from_nii(
    input_root: Union[str, pathlib.Path],
    output_root: Union[str, pathlib.Path],
    nii_file_extension: NiiFileExtension,
    include_ids: list = [-1],
    img_type: str = "png",
) -> None:
    r"""
    Opens nii formated tumor files and saves slices as images

    Args:
        input_root (Union[str, pathlib.Path]): input directory path
        output_root (Union[str, pathlib.Path]):  output directory path
        nii_file_extension (NiiFileExtension): type of nii file
        include_ids (list, optional): converting these given ids.
        img_type (str, optional): output image extension. Defaults to "png".
    """

    if len(include_ids) == 0:  # no ids given
        return

    sample_name = os.path.basename(os.path.normpath(input_root))
    output_path = os.path.join(output_root, sample_name, nii_file_extension.name)

    imgs = nib.load(
        os.path.join(input_root, f"{sample_name}_{nii_file_extension.extension}")
    )

    if include_ids[0] == -1:  # all ids
        include_ids = range(imgs.shape[2])

    imgs = imgs.get_fdata()
    imgs = np.take(imgs, include_ids, axis=2)

    if nii_file_extension != NiiFileExtension.SEG:
        imgs = (imgs / imgs.max()) * 255
        imgs = imgs.astype(np.uint8)
    else:
        tmp = np.zeros(imgs.shape, dtype=np.uint8)
        tmp[np.where(imgs == 1)] = TumorMask.NET.value
        tmp[np.where(imgs == 2)] = TumorMask.ED.value
        tmp[np.where(imgs == 4)] = TumorMask.ET.value
        imgs = tmp

    mkdir(output_root)
    mkdir(output_path)

    for i in range(len(include_ids)):
        filename = os.path.join(output_path, f"{str(i)}.{img_type}")
        img = imgs[:, :, i]
        cv2.imwrite(filename, img)

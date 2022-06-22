import os
import pathlib
from glob import glob
from joblib import Parallel, delayed

try:
    from typing import Literal, Optional, Tuple, Union
except ImportError:
    from typing_extensions import Literal, Optional, Tuple, Union

import numpy as np
import nibabel as nib
import pandas as pd

from src.utils import NiiFileExtension, extract_images_from_nii

import cv2

def preprocess(root_path: str, config: dict) -> None:
    r"""
    Pre-processing module

    Args:
        root_path (str): absolute root path
        config (dict): configuration dictionary

    Necessary elements of config::

        io:
            absolute_paths: bool
            input_path: str
            output_root_path: str
        utility:
            random_seed: int
            num_workers: int
        general:
            dataset_type: "brats18" or "brats20"
            validation_split_ratio: float
        params:
            include_hgg: bool
            include_lgg: bool
            mask_threshold: int or float
            mask_threshold_type: "percentage" or "count"
            step_size: int
            indices_interval: tuple or None
            include_tumor_files: ["t1", "t1ce", "t2", "flair", "segmentation"]
    """

    absolute_paths = config["io"]["absolute_paths"]
    input_path = config["io"]["input_path"]
    output_root_path = config["io"]["output_root_path"]

    random_seed = config["utility"]["random_seed"]
    num_workers = config["utility"]["num_workers"]

    validation_split_ratio = config["general"]["validation_split_ratio"]
    dataset_type = config["general"]["dataset_type"]

    include_hgg = config["params"]["include_hgg"]
    include_lgg = config["params"]["include_lgg"]
    mask_threshold = config["params"]["mask_threshold"]
    mask_threshold_type = config["params"]["mask_threshold_type"]
    step_size = config["params"]["step_size"]
    indices_interval = config["params"]["indices_interval"]
    include_tumor_files = config["params"]["include_tumor_files"]

    if not absolute_paths:
        input_path = os.path.join(root_path, input_path)
        output_root_path = os.path.join(root_path, output_root_path)


    include_lgg = False
    assert (
        include_hgg or include_lgg
    ), "there must be atleast one type of tumor included"

    all_samples = []
    

    dataset_type = dataset_type.lower()

    if dataset_type == "brats20":
    
        df = pd.read_csv(os.path.join(input_path, "name_mapping.csv"))
        if include_hgg:
            all_samples.extend(
                [
                    ("hgg", x)
                    for x in df.loc[
                        df["Grade"] == "HGG", "BraTS_2020_subject_ID"
                    ].tolist()
                ]
            )
        if include_lgg:
            all_samples.extend(
                [
                    ("lgg", x)
                    for x in df.loc[
                        df["Grade"] == "LGG", "BraTS_2020_subject_ID"
                    ].tolist()
                ]
            )
        del df
    elif dataset_type == "brats18":
        
        if include_hgg:
            
            for x in glob(os.path.join(input_path, "hgg/*")):
                
                break
            all_samples.extend(
                [
                    ("hgg", os.path.basename(x))
                    
                    for x in glob(os.path.join(input_path, "hgg/*"))
                ]
            )
        if include_lgg:
            all_samples.extend(
                [
                    ("lgg", os.path.basename(x))
                    for x in glob(os.path.join(input_path, "lgg/*"))
                ]
            )
    else:
        raise ValueError("Given dataset_type in the config is not supported")

    if validation_split_ratio > 0:
        valid_output_root_path = os.path.join(output_root_path, "validation")
        train_output_root_path = os.path.join(output_root_path, "train")

        np.random.seed(random_seed)

        split_index = int(len(all_samples) * validation_split_ratio)
        indices = np.random.permutation(len(all_samples))
        val_ids, train_ids = indices[:split_index], indices[split_index:]
        
        validation = np.array(all_samples)[val_ids, :].tolist()
        training = np.array(all_samples)[train_ids, :].tolist()

        all_samples = [
            (tumor_type, valid_output_root_path, path)
            for tumor_type, path in validation
        ]

        all_samples.extend(
            [
                (tumor_type, train_output_root_path, path)
                for tumor_type, path in training
            ]
        )

        Parallel(n_jobs=num_workers, verbose=10)(
            delayed(extract_tumor_images)(
                input_path=input_path
                if dataset_type == "brats20"
                else os.path.join(input_path, tumor_type.upper()),
                output_root_path=var_output_root_path,
                filename=filename,
                mask_threshold=mask_threshold,
                mask_threshold_type=mask_threshold_type,
                include_tumor_files=include_tumor_files,
                step_size=step_size,
                indices_interval=indices_interval,
                tumor_type=tumor_type,
            )
            for (tumor_type, var_output_root_path, filename) in all_samples
        )
    else:
        Parallel(n_jobs=num_workers, verbose=10)(
            delayed(extract_tumor_images)(
                input_path=input_path
                if dataset_type == "brats20"
                else os.path.join(input_path, tumor_type.upper()),
                output_root_path=output_root_path,
                filename=filename,
                mask_threshold=mask_threshold,
                mask_threshold_type=mask_threshold_type,
                include_tumor_files=include_tumor_files,
                step_size=step_size,
                indices_interval=indices_interval,
                tumor_type=tumor_type,
            )
            for tumor_type, filename in all_samples
        )


def extract_image_ids(
    input_path: Union[str, pathlib.Path],
    filename: str,
    step_size: int = 1,
    mask_threshold_type: Optional[
        Union[Literal["percentage"], Literal["count"]]
    ] = None,
    mask_threshold: Optional[Union[float, int]] = None,
    indices_interval: Tuple[int] = None,
) -> list:
    r"""
    Extracts ids of images if they are bigger than given mask threshold

    Args:
        input_path (Union[str, pathlib.Path]): path of the input images
        filename (str): name of the sample
        step_size (int): step size.
        mask_threshold_type (str): type of treshold to be applied to masks.
            percentage as minimum ratio of mask to whole image
            count as minimum number of pixels in mask
        mask_threshold (int, optional): threshold value. Defaults to -1.
        indices_interval (Tuple[int]): specifically check given indices.
    """

    segs = nib.load(
        os.path.join(input_path, f"{filename}_{NiiFileExtension.SEG.extension}")
    )

    segs = segs.get_fdata()
    #print(segs.shape)
    
    #segs = []
    #for img in glob(os.path.join(input_path, f"{filename}_seg", "*.png")):
    #    n= cv2.imread(img)
    #    segs.append(n)
    #segs = np.array(segs)
    #print(segs.shape)
    

    if indices_interval is not None:
        assert (
            indices_interval[0] <= indices_interval[1]
            and indices_interval[0] >= 0
            and indices_interval[1] >= 0
            and indices_interval[0] <= segs.shape[2] - 1
            and indices_interval[1] <= segs.shape[2] - 1
        )

    if mask_threshold_type is None:
        if indices_interval is None:
            return range(segs.shape[2])
        else:
            return range(indices_interval[0], indices_interval[1] + 1, step_size)
    else:
        if mask_threshold_type == "percentage":
            assert mask_threshold is not None or (
                mask_threshold >= 0.0 and mask_threshold <= 1.0
            ), "Mask threshold must given a percentage float value"
        elif mask_threshold_type == "count":
            assert mask_threshold is not None or (
                mask_threshold is int and mask_threshold >= 0
            ), "Mask threshold must given a countable integer value"

    include_ids = []
    for i in range(0, segs.shape[2], step_size):
        tmp = segs[:, :, i]
        if mask_threshold_type == "count":
            if (tmp > 0).sum() > mask_threshold:
                include_ids.append(i)
        elif mask_threshold_type == "percentage":
            ratio = (tmp > 0).sum() / tmp.size
            if ratio > mask_threshold:
                include_ids.append(i)
    return include_ids


def extract_tumor_images(
    input_path: Union[str, pathlib.Path],
    output_root_path: Union[str, pathlib.Path],
    filename: str,
    mask_threshold: Union[int, float],
    mask_threshold_type: Union[Literal["percentage"], Literal["count"]],
    include_tumor_files: List[str],
    step_size: int = 1,
    indices_interval: Optional[Tuple[int, int]] = None,
    tumor_type: str = "",
) -> None:
    r"""
    Extracts images from tumor files

    Args:
        input_path (Union[str, pathlib.Path]): input directory path.
        output_root_path (Union[str, pathlib.Path]): output directory path.
        filename (str): filename or sample name.
        mask_threshold (int or float): threshold value.
        mask_threshold_type (str): type of the threshold "percentage" or "count".
        include_tumor_files (list[str]): list of tumor files to be preprocessed.
        step_size (int): step size for indices.
        tumor_type (str): "hgg", "lgg".
    """

    include_ids = [-1]  # default: include all
    include_ids = extract_image_ids(
        input_path=os.path.join(input_path, filename),
        filename=filename,
        step_size=step_size,
        mask_threshold=mask_threshold,
        mask_threshold_type=mask_threshold_type,
        indices_interval=indices_interval,
    )

    input_root = os.path.join(input_path, filename)
    if tumor_type:
        output_root = os.path.join(output_root_path, tumor_type)
    else:
        output_root = output_root_path

    if "segmentation" in include_tumor_files:
        extract_images_from_nii(
            input_root=input_root,
            output_root=output_root,
            nii_file_extension=NiiFileExtension.SEG,
            img_type="png",
            include_ids=include_ids,
        )
    if "t1" in include_tumor_files:
        extract_images_from_nii(
            input_root=input_root,
            output_root=output_root,
            nii_file_extension=NiiFileExtension.T1,
            img_type="png",
            include_ids=include_ids,
        )
    if "t2" in include_tumor_files:
        extract_images_from_nii(
            input_root=input_root,
            output_root=output_root,
            nii_file_extension=NiiFileExtension.T2,
            img_type="png",
            include_ids=include_ids,
        )
    if "t1ce" in include_tumor_files:
        extract_images_from_nii(
            input_root=input_root,
            output_root=output_root,
            nii_file_extension=NiiFileExtension.T1CE,
            img_type="png",
            include_ids=include_ids,
        )
    if "flair" in include_tumor_files:
        extract_images_from_nii(
            input_root=input_root,
            output_root=output_root,
            nii_file_extension=NiiFileExtension.FLAIR,
            img_type="png",
            include_ids=include_ids,
        )

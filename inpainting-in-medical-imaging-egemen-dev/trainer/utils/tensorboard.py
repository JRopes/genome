from functools import reduce
from typing import Dict, Optional
import os
import io

from trainer.typing import PathType

from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd


def extract_tb_scalars(
    path: PathType, save: bool = False, save_name: str = "scalars.csv"
) -> Optional[pd.DataFrame]:
    r"""
    Extract scalars values from tensorboard event file and returns them as Dataframe

    Args:
        path (PathType): path where the event file is located.
        save (bool): save the dataframe as a csv file.
        save_name (str): name of a csv file
    """
    event = EventAccumulator(path=path)
    event.Reload()
    tags = event.Tags()["scalars"]

    if len(tags) == 0:
        return None

    dfs = []
    for tag in tags:
        dfs.append(
            pd.DataFrame(event.Scalars(tag))
            .rename(columns={"value": tag})
            .drop(["wall_time"], axis=1)
        )
    out = reduce(
        lambda left, right: pd.merge(left, right, on=["step"], how="outer"), dfs
    )
    if save:
        out.to_csv(os.path.join(path, save_name), index=False)
    return out


def extract_tb_images(
    path: PathType, save: bool = False, image_folder_name: str = "images"
) -> Optional[Dict[str, Dict[int, Image.Image]]]:
    r"""
    Extract images from tensorboard event file and returns them as dictionary of PIL images

    Args:
        path (PathType): path where the event file is located.
        save (bool): save the images in a folder
        save_name (str): name of a folder
    """
    event = EventAccumulator(path=path, size_guidance={"images": 0})
    event.Reload()
    tags = event.Tags()["images"]

    if len(tags) == 0:
        return None

    if save:
        if not os.path.exists(os.path.join(path, image_folder_name)):
            os.makedirs(os.path.join(path, image_folder_name))

    images = {}

    for tag in tags:
        images[tag] = {}
        for item in event.Images(tag):
            step = item.step
            encoded = item.encoded_image_string
            img = Image.open(io.BytesIO(encoded))
            if save:
                output_path = os.path.join(path, image_folder_name, str(step))
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                img.save(os.path.join(output_path, f"{tag.replace('/', '_')}.png",))

            images[tag][step] = img

    return images

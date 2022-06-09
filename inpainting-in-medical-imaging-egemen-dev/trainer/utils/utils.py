import os
from pathlib import Path
from datetime import datetime
from typing import Literal, Optional, Union

from trainer.typing import DeviceType

import torch


def get_current_timestamp(fmt: str = "%d-%m-%Y %H-%M-%S") -> str:
    r"""
    Get the current timestamp in string with given format

    Args:
        fmt (str): time string format.

    Example:
    >>> get_current_timestamp("%d-%m-%Y %H-%M-%S")
    '08-01-2022 19-28-49'
    """
    return datetime.now().strftime(fmt)


def create_path_with_timestamp(
    root: Optional[Union[str, Path]] = None,
    prefix: Optional[str] = None,
    postfix: Optional[str] = None,
) -> str:
    r"""
    Create a path with current timestamp by calling `get_current_timestamp` function within.

    Args:
        root (str, Path): root path.
        prefix (str): string to be added before the timestamp.
        postfix (str): string to be added after the timestamp.

    Example:
    >>> create_path_with_timestamp(prefix="pre")
    'pre 08-01-2022 19-28-49'
    >>> create_path_with_timestamp(prefix="pre", postfix="post")
    'pre 08-01-2022 19-28-49 post'
    >>> create_path_with_timestamp()
    '08-01-2022 19-28-49'
    """
    out, name = [], None
    if root:
        out.append(root)
    if prefix:
        name = prefix
    if name is not None:
        name += " " + get_current_timestamp()
    else:
        name = get_current_timestamp()
    if postfix:
        name += " " + postfix
    out.append(name)
    return os.path.join(*out)


def get_device() -> DeviceType:
    r"""
    Get available device
    """
    return (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if torch.cuda.is_available()
        else "cpu"
    )


class StageContext:
    def __init__(self, ref_object, stage: str):
        self._ref_object = ref_object
        self._prev_stage = ref_object._stage
        self._stage = stage

    def __enter__(self):
        self._ref_object._stage = self._stage

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            return False
        self._ref_object._stage = self._prev_stage
        return True

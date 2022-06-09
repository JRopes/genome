from typing import Literal, Union
from pathlib import Path

import torch
from tqdm import tqdm

from trainer.utils import ProgressBar

PBarType = Union[tqdm, ProgressBar]
PathType = Union[str, Path]
TBLevelType = Union[Literal["auto"], Literal["manual"], Literal["none"]]
DeviceType = Union[torch.device, Literal["cpu"]]

import os
import random
import torch
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """
    Makes necessary adjustments for reproducibility

    Args:
        seed (int): random number generator seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

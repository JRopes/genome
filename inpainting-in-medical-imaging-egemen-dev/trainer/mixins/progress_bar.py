from typing import Union

from tqdm import tqdm


from trainer.utils import Stage, ProgressBar
from trainer.typing import PBarType


class ProgressBarMixin:
    r"""
    Mixin class for progress bar handling
    """

    def _create_progressbar(self) -> PBarType:
        r"""
        Creates a progress bar
        """
        if self._stage == Stage.TRAINING:
            loader = self._training_loader
            desc = "Training"
        elif self._stage == Stage.VALIDATION:
            loader = self._validation_loader
            desc = "Validation"
        else:
            raise ValueError(
                f"Current stage {self._stage} is not appropriate for progress bar creation"
            )
        return (
            tqdm(loader, desc=f"[{self._curr_epoch}] {desc} ", ncols=0)
            if self._using_tqdm
            else ProgressBar(loader, self._curr_epoch, desc)
        )

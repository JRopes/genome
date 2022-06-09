from typing import Literal, Union
from trainer.utils import extract_tb_scalars, extract_tb_images
from trainer.utils import Stage


class TensorboardMixin:
    r"""
    Mixin class for tensorboard functionalities
    """

    def _save_scalars(self, file_name: str = "scalars.csv") -> None:
        r"""
        Saves scalars metrics from tensorboard to csv

        Args:
            file_name (str): name of the file to save scalars
        """
        if self._tb_level != "none":
            extract_tb_scalars(self._output_path, True, file_name)

    def _save_images(self, folder_name: str = "images") -> None:
        r"""
        Saves images from tensorboard as png files

        Args:
            folder_name (str): name of the folder to save images
        """
        if self._tb_level != "none":
            extract_tb_images(self._output_path, True, folder_name)

    def _auto_transfer_scalars(
        self, category: Union[Literal["batch"], Literal["epoch"]]
    ):
        r"""
        Transfers all metric values to tensorboard if `tb_level` parameter is defined as 'auto'

        Args:
            category (str): name of the metric category to transfer
        """
        if self._tb_level == "auto":
            if self._stage == Stage.TRAINING:
                vals = self.metrics.stats["train"][category]
                if vals is not None:
                    for k, v in vals.items():
                        if isinstance(v, (float, int)):
                            self.tb_writer.add_scalar(
                                f"train/epoch_{k}", v, self._curr_epoch
                            )
            elif self._stage == Stage.VALIDATION:
                vals = self.metrics.stats["val"][category]
                if vals is not None:
                    for k, v in vals.items():
                        if isinstance(v, (float, int)):
                            self.tb_writer.add_scalar(
                                f"val/epoch_{k}", v, self._curr_epoch
                            )

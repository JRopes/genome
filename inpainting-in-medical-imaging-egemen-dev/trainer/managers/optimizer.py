from typing import Optional

from torch.optim import Optimizer

from .base import BaseTorchModuleManager


class OptimizerManager(BaseTorchModuleManager):
    def __init__(self) -> None:
        r"""
        Manager for pytorch optimizers
        """
        super().__init__()
        self._additional_info = dict()

    def get(self, id: str) -> Optimizer:
        r"""
        Get an optimizer with a given id

        Args:
            id (str): identifer of optimizer
        """
        return self._instances[id]

    def add(
        self,
        id: str,
        optimizer: Optimizer,
        allow_save: bool = True,
        allow_load: bool = True,
        allow_stats: bool = True,
        additional_info: Optional[dict] = None,
    ) -> None:
        r"""
        Add an optimizer

        Args:
            id (str): identifer of optimizer
            optimizer (nn.Module): optimizer instance
            additional_info (dict): info to be added when `stats` function is called
        """
        if id in self._instances:
            raise ValueError(f"Given id '{id}' is already registered")
        self._instances[id] = optimizer
        self._set_cp_details(id, "optimizer", allow_save, allow_load, allow_stats)
        if additional_info:
            self._additional_info[id] = additional_info

    @property
    def stats(self) -> dict:
        r"""
        Return a dictionary of stats/info
        """
        out = {}
        for id, opt in self._instances.items():
            out[id] = {}
            if id in self._additional_info:
                out[id]["additional_info"] = self._additional_info[id]
        return out

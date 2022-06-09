from abc import abstractmethod
from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Union
from abc import ABC
from pathlib import Path

import torch


class BaseTorchModuleManager(ABC):
    @dataclass(init=True, repr=True, frozen=True)
    class CheckpointDetails:
        instance_type_name: str = None
        allow_save: bool = True
        allow_load: bool = True
        allow_stats: bool = True

    def __init__(self) -> None:
        r"""
        Base Manager
        """
        self._instances: Dict[str, Any] = dict()
        self._checkpoint_details: Dict[str, self.CheckpointDetails] = dict()

    def _set_cp_details(
        self,
        id: str,
        instance_type_name: str,
        allow_save: bool,
        allow_load: bool,
        allow_stats: bool,
    ) -> None:
        r"""
        Helper function to set checkpoint details for corresponding instance

        Args:
            id (str): identifier of instance
            instance_type_name (str): prefix for the filename.
            allow_save (bool): whether to save instance as checkpoint.
            allow_load (bool): whether to load instance from checkpoint.
            allow_stats (bool): whether to add stats/info when `stats` function is called.
        """
        detail = self.CheckpointDetails(
            instance_type_name, allow_save, allow_load, allow_stats
        )
        self._checkpoint_details[id] = detail

    def load(self, root: Optional[Union[str, Path]]) -> int:
        r"""
        Loads all loadable instances from given root path

        Args:
            root(str or Path): root path to load checkpoints from.
        """
        max_epoch = 0
        for id, instance in self._instances.items():
            cp_details = self._checkpoint_details[id]
            if cp_details.allow_load:
                if cp_details.instance_type_name is not None:
                    filename = f"{cp_details.instance_type_name}_{id}.pth.tar"
                else:
                    filename = f"{id}.pth.tar"
                path = os.path.join(root, filename)
                cp = torch.load(path)
                instance.load_state_dict(cp["state_dict"])
                if "epoch" in cp:
                    max_epoch = max(max_epoch, cp["epoch"])
        return max_epoch

    def save(self, root: Union[str, Path], epoch: int) -> None:
        r"""
        Loads all loadable instances from given root path

        Args:
            root(str or Path): root path to load checkpoints from.
            epoch (int): current epoch.
        """
        for id, instance in self._instances.items():
            cp_details = self._checkpoint_details[id]
            if cp_details.allow_save:
                if cp_details.instance_type_name is not None:
                    filename = f"{cp_details.instance_type_name}_{id}.pth.tar"
                else:
                    filename = f"{id}.pth.tar"
                path = os.path.join(root, filename)
                torch.save({"state_dict": instance.state_dict(), "epoch": epoch}, path)

    @abstractmethod
    def get(self, *args, **kwargs) -> Any:
        r"""
        Getter function
        """
        ...

    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        r"""
        Setter function
        """
        ...

    @property
    @abstractmethod
    def stats(self) -> dict:
        r"""
        Returns dictionary of stats/info
        """
        ...

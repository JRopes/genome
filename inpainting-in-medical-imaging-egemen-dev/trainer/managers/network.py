from typing import Optional
import torch.nn as nn

from .base import BaseTorchModuleManager


class NetworkManager(BaseTorchModuleManager):
    def __init__(self) -> None:
        r"""
        Manager for pytorch networks
        """
        super().__init__()
        self._additional_info = dict()

    def get(self, id: str) -> nn.Module:
        r"""
        Returns a network with a given id

        Args:
            id (str): identifer of network
        """
        return self._instances[id]

    def get_parameters(self, id: str) -> dict:
        r"""
        Returns network's parameters

        Args:
            id (str): identifer of network
        """
        return self._instances[id].parameters()

    def add(
        self,
        id: str,
        network: nn.Module,
        allow_save: bool = True,
        allow_load: bool = True,
        allow_stats: bool = True,
        additional_info: Optional[dict] = None,
    ) -> None:
        r"""
        Add a network

        Args:
            id (str): identifer of network
            network (nn.Module): nn.Module instance
            additional_info (dict): info to be added when :func:`stats` is called
        """
        if id in self._instances:
            raise ValueError(f"Given id '{id}' is already registered")
        self._instances[id] = network
        self._set_cp_details(id, "network", allow_save, allow_load, allow_stats)
        if additional_info:
            self._additional_info[id] = additional_info

    @property
    def stats(self) -> dict:
        r"""
        Returns a dictionary of stats/info
        """
        out = {}
        for id, net in self._instances.items():
            out[id] = {}
            total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            out[id]["total_params"] = total_params
            # modules = []
            # for _, m in enumerate(net.modules()):
            #     modules.append(str(m))
            # out[id]["modules"] = modules
            if id in self._additional_info:
                out[id].update(self._additional_info[id])
        return out

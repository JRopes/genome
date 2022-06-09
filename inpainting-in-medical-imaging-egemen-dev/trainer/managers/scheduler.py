from typing import Literal, Optional, Union

from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from .base import BaseTorchModuleManager
from .metric import MetricManager


class LRSchedulerManager(BaseTorchModuleManager):
    def __init__(self, metrics: MetricManager) -> None:
        r"""
        Manager for learning rate schedulers

        Args:
            metrics (MetricManager): metric manager instance
        """
        super().__init__()
        self._target_metrics = {
            "train": {"batch": {}, "epoch": {}},
            "val": {"batch": {}, "epoch": {}},
        }
        self._metrics = metrics
        self._additional_info = dict()

    def get(self, id: str) -> LRScheduler:
        r"""
        Returns a LR scheduler with a given id

        Args:
            id (str): identifer of scheduler
        """
        return self._instances[id]

    def add(
        self,
        id: str,
        scheduler: LRScheduler,
        metric_id: str,
        metric_category: Union[Literal["batch"], Literal["epoch"]] = "epoch",
        metric_stage: Union[Literal["train"], Literal["val"]] = "val",
        allow_save: bool = True,
        allow_load: bool = True,
        allow_stats: bool = True,
        additional_info: Optional[dict] = None,
    ) -> None:
        r"""
        Add a LR scheduler

        Args:
            id (str): identifer of scheduler
            scheduler (LRScheduler): LRScheduler instance
            metric_id (str): identifier of a target metric which will be used for stepping
            metric_category (str): category of a metric
            metric_stage (str): stage of a metric
            additional_info (dict): info to be added when :func:`stats` is called

        Examples:

        >>> metrics = MetricsManager()
        >>> manager = LRSchedulerManager(metrics)
        >>> ...
        >>> lr_scheduler = ReduceLROnPlateau(...)
        >>> manager.add(lr_scheduler, "metric", "epoch", "val")

        """
        if id in self._instances:
            raise ValueError(f"Given id '{id}' is already registered")
        self._instances[id] = scheduler
        self._target_metrics[metric_stage][metric_category][id] = metric_id
        self._set_cp_details(id, "scheduler", allow_save, allow_load, allow_stats)
        if additional_info:
            self._additional_info[id] = additional_info

    @property
    def stats(self) -> dict:
        r"""
        Returns a dictionary of stats/info
        """
        out = {}
        for stage in ["train", "val"]:
            for category in ["batch", "epoch"]:
                for id, t_id in self._target_metrics[stage][category].items():
                    out[id] = {
                        "target_metric": {
                            "id": t_id,
                            "category": category,
                            "stage": stage,
                        },
                        # **self._instances[id].state_dict,
                    }
                    if id in self._additional_info:
                        out[id]["additional_info"] = self._additional_info[id]

        return out

    def step_all(self) -> None:
        r"""
        Runs the `step` function for all the schedulers

        TODO currently is being called after the epochs,
            needs to be divided into different parts: batch and epoch
            or schedulers will only work after the epochs not batches
        """
        for stage in ["train", "val"]:
            for category in ["batch", "epoch"]:
                for id, t_id in self._target_metrics[stage][category].items():
                    self._instances[id].step(self._metrics.get(t_id, category, stage))

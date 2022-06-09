from typing import Any, Dict, Literal, Optional, Union
from collections import defaultdict

import numpy as np


class CumulativeMetric:
    def __init__(self, value: Optional[Any] = None) -> None:
        r"""
        Cumulation value holder

        Args:
            value (Any,optional): initial value
        """
        self.value = value
        self.iter_add_count = 1

    @property
    def normalized(self) -> Any:
        r"""
        Normalizes by dividing the value with `iter_add_count`
        """
        if self.value is not None:
            return self.value / self.iter_add_count
        else:
            raise ValueError("Metric value must be initialized before normalization.")

    def reset(self):
        r"""
        Reset the value and count
        """
        self.value = None
        self.iter_add_count = 1

    def add(self, other: Any):
        if self.value is not None:
            if isinstance(self.value, type(other)):
                if self.value is None:
                    self.value = other
                else:
                    self.value += other
                    self.iter_add_count += 1
            else:
                raise TypeError(
                    f"Expected '{type(self.value)}' but got '{type(other)}'."
                )
        else:
            self.value = other


class MetricManager:
    def __init__(self) -> None:
        r"""
        Manager to handle metrtics automatically by accumulating over batches
        and normalizing when epoch is finished
        """
        self._cumulation_train = defaultdict(CumulativeMetric)
        self._cumulation_val = defaultdict(CumulativeMetric)

        self._batch_train = dict()
        self._batch_val = dict()

        self._epoch_train = dict()
        self._epoch_val = dict()

    def cumulate(
        self,
        metric_dict: Dict[str, Any],
        stage: Union[Literal["train"], Literal["val"]],
    ) -> None:
        r"""
        Cumulates the metric values

        Args:
            metric_dict (dict): dictionary with the metric names and values
            stage (str): stage of the metrics.
        """
        if stage == "train":
            for k, v in metric_dict.items():
                self._batch_train[k] = v
                if k not in self._cumulation_train:
                    self._cumulation_train[k] = CumulativeMetric(v)
                else:
                    self._cumulation_train[k].add(v)
        elif stage == "val":
            for k, v in metric_dict.items():
                self._batch_val[k] = v
                if k not in self._cumulation_val:
                    self._cumulation_val[k] = CumulativeMetric(v)
                else:
                    self._cumulation_val[k].add(v)
        else:
            raise ValueError(f"Invalid stage value: {stage}")

    def normalize(
        self, stage: Union[Literal["train"], Literal["val"]], reset: bool = True
    ) -> None:
        r"""
        Normalizes cumulations and assigns them to epoch metric values

        Args:
            stage (str): stage of the metric.
            reset (bool): reset the cumulations
        """
        if stage == "train":
            for k in self._cumulation_train.keys():
                self._epoch_train[k] = self._cumulation_train[k].normalized
                if reset:
                    self._cumulation_train[k].reset()
        elif stage == "val":
            for k in self._cumulation_val.keys():
                self._epoch_val[k] = self._cumulation_val[k].normalized
                if reset:
                    self._cumulation_val[k].reset()
        else:
            raise ValueError(f"Invalid stage value: {stage}")

    def get(
        self,
        id: str,
        category: Union[Literal["batch"], Literal["epoch"]],
        stage: Union[Literal["train"], Literal["val"]],
    ) -> Any:
        r"""
        Manual getter function for the metric

        Args:
            id (str): identifier of the metric.
            category (str): batch or epoch
            stage (str): stage of the metric.
        """
        if stage == "train":
            if category == "epoch":
                assert id in self._epoch_train, "identifier does not exist"
                return self._epoch_train[id]
            elif category == "batch":
                assert id in self._batch_train, "identifier does not exist"
                return self._batch_train[id]
            else:
                raise ValueError(f"Invalid category value: {category}")
        elif stage == "val":
            if category == "epoch":
                assert id in self._epoch_val, "identifier does not exist"
                return self._epoch_val[id]
            elif category == "batch":
                assert id in self._batch_val, "identifier does not exist"
                return self._batch_val[id]
            else:
                raise ValueError(f"Invalid category value: {category}")
        else:
            raise ValueError(f"Invalid stage value: {stage}")

    def set(
        self,
        id: str,
        value: Any,
        category: Union[Literal["batch"], Literal["epoch"]],
        stage: Union[Literal["train"], Literal["val"]],
    ) -> None:
        r"""
        Manual setter function for the metric

        Args:
            id (str): identifier of the metric.
            category (str): batch or epoch
            stage (str): stage of the metric.
        """
        if stage == "train":
            if category == "epoch":
                self._epoch_train[id] = value
            elif category == "batch":
                self._batch_train[id] = value
            else:
                raise ValueError(f"Invalid category value: {category}")
        elif stage == "val":
            if category == "epoch":
                self._epoch_val[id] = value
            elif category == "batch":
                self._batch_val[id] = value
            else:
                raise ValueError(f"Invalid category value: {category}")
        else:
            raise ValueError(f"Invalid stage value: {stage}")

    @property
    def stats(self) -> Dict[str, Dict[str, Any]]:
        r"""
        Returns final metric values
        """
        return {
            "train": {"epoch": self.parse_metrics(self._epoch_train)},
            "val": {"epoch": self.parse_metrics(self._epoch_val)},
        }

    @staticmethod
    def parse_metrics(d: dict) -> dict:
        r"""
        Helper parser function to distinguish lists from numericals while transferring
        them to proper 'stats' which will then be written to an output file
        """
        out = {}
        for k, v in d.items():
            if isinstance(v, (np.ndarray, np.generic)):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out

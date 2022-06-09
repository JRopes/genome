import os
from pathlib import Path
from typing import Optional
import yaml


class CheckpointMixin:
    r"""
    Mixin class for checkpoints and general stats
    """

    def _save_checkpoints(self, reason: Optional[str] = None) -> None:
        r"""
        Auto saving mechanism for networks, optimizers and schedulers

        Args:
            reason (str, optional): if given, will be used as a folder name instead of epoch number
        """
        if reason:
            path = Path(self._output_path, "checkpoints", reason)
        else:
            path = Path(self._output_path, "checkpoints", str(self._curr_epoch))

        path.mkdir(parents=True)
        self.networks.save(path, self._curr_epoch)
        self.optimizers.save(path, self._curr_epoch)
        self.schedulers.save(path, self._curr_epoch)

    def _load_checkpoints(self) -> None:
        r"""
        Auto loading mechanism for networks, optimizers and schedulers
        """
        print(f">>> Resuming: loading checkpoint(s) from {self._resume_path}")
        self._start_epoch = (
            max(
                self.networks.load(self._resume_path),
                self.optimizers.load(self._resume_path),
                self.schedulers.load(self._resume_path),
            )
            + 1
        )
        self._end_epoch += self._start_epoch - 1

    def _save_stats(self):
        r"""
        Saves stats
        """
        if self._resume_path is not None:
            self.stats["resumed_from"] = str(self._resume_path)
        self.stats["starting_epoch"] = self._start_epoch
        self.stats["ending_epoch"] = self._end_epoch
        self.stats["last_current_epoch"] = self._curr_epoch
        self.stats["number_of_batches_train"] = self._train_n_batches
        self.stats["number_of_batches_val"] = self._val_n_batches
        self.stats["mixed_precision"] = self._using_mixed_precision
        self.stats["benchmark"] = self._using_benchmark
        self.stats["deterministic"] = self._using_deterministic
        self.stats["networks"] = self.networks.stats
        self.stats["metrics"] = self.metrics.stats
        # self.stats["optimizers"] = self.optimizers.stats
        # self.stats["schedulers"] = self.schedulers.stats
        with open(os.path.join(self._output_path, "stats.yml"), "w",) as f:
            yaml.dump(
                self.stats, f, default_flow_style=False, sort_keys=False, width=4096
            )

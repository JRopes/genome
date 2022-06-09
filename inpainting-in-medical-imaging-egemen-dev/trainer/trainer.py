import os
from typing import Any, Dict, Optional
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from trainer.mixins import (
    CallbackMixin,
    CheckpointMixin,
    TensorboardMixin,
    ProgressBarMixin,
)
from trainer.managers import (
    MetricManager,
    NetworkManager,
    OptimizerManager,
    LRSchedulerManager,
)
from trainer.utils import StageContext, Stage
from trainer.utils import create_path_with_timestamp, get_device, make_archive
from trainer.typing import PBarType, PathType, TBLevelType, DeviceType


class BaseTrainer(CheckpointMixin, CallbackMixin, TensorboardMixin, ProgressBarMixin):
    def __init__(
        self,
        training_loader: Optional[DataLoader] = None,
        validation_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        checkpoint_save_gap: int = 5,
        params: Optional[Dict[str, Any]] = None,
        output_path: Optional[PathType] = None,
        resume_path: Optional[PathType] = None,
        mixed_precision: bool = False,
        benchmark: bool = True,
        deterministic: bool = False,
        tensorboard_level: TBLevelType = "auto",
        tqdm_progress_bar: bool = True,
        zip_at_the_end: bool = False,
    ) -> None:
        r"""

        Base class for trainers

        Args:
            training_loader (DataLoader, optional): dataloader instance for training data.
            validation_loader (DataLoader, optional): dataloader instance for validation data.
            epochs (int): number of epochs to run.
            checkpoint_save_gap (bool): if bigger than 0, a checkpoint will be saved at each gap, otherwise no saving.
            params (dict, optional): custom dictionary in which user-defined items exist.
            output_path (str or Path, optional): output path to dump the results.
            resume_path (str or Path, optional): if given, trainer will continue from checkpoints located in path.
            mixed_precision (bool):  mixed precision flag for PyTorch but not automatic implementation-wise.
            benchmark (bool): enables cuDNN to benchmark multiple convolution algorithms and select the fastest.
            deterministic (bool): enables cuDNN to only use deterministic convolution algorithms.
            tensorboard_level (str): level of tensorboard involvement; 'auto' activates auto transfer of scalars,
              'manual' only allows manual registers from user in derived trainer class, 'none' disables tensorboard
            tqdm_progress_bar (bool): enables tqdm progress bar.
            zip_at_the_end (bool): zip the entire training/validation folder

        Examples::

            class ModelTrainer(BaseTrainer):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.network = Net()
                    self.criterion = nn.MSELoss()
                    self.optimizer = optim.SGD(self.net.parameters(), lr=0.01)
                def training_step(self, progress_bar, batch):
                    x, target = batch
                    self.optimizer.zero_grad()
                    output = self.network(x)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    return { "loss" : loss }
                def validation_step(self, progress_bar,  batch):
                    x, target = batch
                    output = self.network(x)
                    metric = self.criterion(output, target)
                    return { "metric" : metric }
        """

        assert (
            training_loader is not None or validation_loader is not None
        ), "Either training_loader or validation_loader must be specified."

        if training_loader is not None:
            assert isinstance(training_loader, DataLoader)

        if validation_loader is not None:
            assert isinstance(validation_loader, DataLoader)

        assert (
            isinstance(epochs, int) and epochs > 0
        ), "Epochs parameter must be an integer bigger than 0."

        assert isinstance(mixed_precision, bool)
        assert isinstance(benchmark, bool)
        assert isinstance(deterministic, bool)
        assert isinstance(tqdm_progress_bar, bool)
        assert isinstance(zip_at_the_end, bool)
        assert isinstance(checkpoint_save_gap, int)

        # Internal instance variables

        self._resume = False
        self._resume_path = None
        if resume_path is not None:
            self._resume_path = Path(resume_path)
            assert os.path.isdir(
                resume_path
            ), "Given resume_path is not a valid directory"
            self._resume = True

        if output_path is not None:
            self._output_path = create_path_with_timestamp(output_path)
        else:
            self._output_path = ""

        self._stage = Stage.INTERMISSION
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic

        self._device = get_device()

        self._start_epoch = 1
        self._curr_epoch = -1

        self._curr_train_batch_n = 0
        self._curr_val_batch_n = 0

        self._end_epoch = epochs
        self._checkpoint_save_gap = checkpoint_save_gap

        self._training_loader = training_loader
        self._validation_loader = validation_loader

        self._train_n_batches = (
            len(self._training_loader) if self._training_loader is not None else 0
        )
        self._val_n_batches = (
            len(self._validation_loader) if self._validation_loader is not None else 0
        )

        self._using_mixed_precision = mixed_precision
        self._using_benchmark = benchmark
        self._using_deterministic = deterministic
        self._using_tqdm = tqdm_progress_bar
        self._tb_level = tensorboard_level
        self._zip_at_the_end = zip_at_the_end

        self._terminate = False
        self._force_save_checkpoint = None

        # User accessible instance variables

        self.params = params
        self.metrics = MetricManager()
        self.networks = NetworkManager()
        self.optimizers = OptimizerManager()
        self.schedulers = LRSchedulerManager(self.metrics)
        self.tb_writer = (
            SummaryWriter(self._output_path) if self._tb_level != "none" else None
        )
        self.callbacks = []
        self.stats = {}

    @property
    def device(self) -> DeviceType:
        r"""
        Returns the identifier of device that is being used
        """
        return self._device

    @property
    def current_epoch(self) -> int:
        r"""
        Returns the current epoch
        """
        return self._curr_epoch

    @property
    def on_last_epoch(self) -> bool:
        r"""
        Boolean of whether the current epoch is the last epoch
        """
        return self._curr_epoch == self._end_epoch

    @property
    def on_saving_epoch(self) -> bool:
        r"""
        Boolean of whether the current epoch is the saving epoch
        """
        return self._curr_epoch % self._checkpoint_save_gap == 0

    @property
    def on_last_batch(self) -> bool:
        r"""
        Boolean of whether the current batch is the last batch of the epoch
        """
        if self._stage == Stage.TRAINING:
            return self._curr_train_batch_n == self._train_n_batches - 1
        elif self._stage == Stage.VALIDATION:
            return self._curr_val_batch_n == self._val_n_batches - 1
        else:
            return False

    @property
    def amp_enabled(self) -> bool:
        r"""
        Boolean of whether automatic mixed precision is being used
        """
        return self._using_mixed_precision

    @property
    def tensorboard_enabled(self) -> bool:
        r"""
        Boolean of whether tensorboard is being used
        """
        return self._tb_level

    @property
    def training_enabled(self) -> bool:
        r"""
        Boolean of whether training loader is present and being used
        """
        return self._train_n_batches != 0

    @property
    def validation_enabled(self) -> bool:
        r"""
        Boolean of whether validation loader is present and being used
        """
        return self._val_n_batches != 0

    def start(self) -> None:
        r"""
        Start trainer
        """
        self._main()

    def training_step(self, progress_bar: PBarType, batch: Any) -> Dict[str, Any]:
        r"""
        Main training step function within the epoch loop

        Args:
            progress_bar (tqdm.tqdm): progress bar.
            batch (Any): current training batch retrieved from dataloader.
        """
        ...

    def validation_step(self, progress_bar: PBarType, batch: Any) -> Dict[str, Any]:
        r"""
        Main validation step function within the epoch loop

        Args:
            progress_bar (tqdm.tqdm): progress bar.
            batch (Any): current validation batch retrieved from dataloader.
        """
        ...

    def on_training_batch_start(self, progress_bar: PBarType, batch: Any) -> None:
        r"""
        Positonal function that runs at the beginning for every training batch

        Args:
            progress_bar (tqdm.tqdm): progress bar.
            batch (Any): current training batch retrieved from dataloader.
        """
        ...

    def on_training_batch_end(
        self, progress_bar: PBarType, batch: Any, output: Dict[str, Any]
    ) -> None:
        r"""
        Positonal function that runs at the end for every training batch

        Args:
            progress_bar (tqdm.tqdm): progress bar.
            batch (Any): current training batch retrieved from dataloader.
            output (dict): current output results from training step
        """
        ...

    def on_validation_batch_start(self, progress_bar: PBarType, batch: Any) -> None:
        r"""
        Positonal function that runs at the beginning for every validation batch

        Args:
            progress_bar (tqdm.tqdm): progress bar.
            batch (Any): current validation batch retrieved from dataloader.
        """
        ...

    def on_validation_batch_end(
        self, progress_bar: PBarType, batch: Any, output: Dict[str, Any]
    ) -> None:
        r"""
        Positonal function that runs at the end for every validation epoch

        Args:
            progress_bar (tqdm.tqdm): progress bar.
            batch (Any): current validation batch retrieved from dataloader.
            output (dict): current output results from validation step
        """
        ...

    def on_training_epoch_start(self) -> None:
        r"""
        Positonal function that runs at the beginning for every training epoch
        """
        ...

    def on_training_epoch_end(self) -> None:
        r"""
        Positonal function that runs at the end for every training epoch
        """
        ...

    def on_validation_epoch_start(self) -> None:
        r"""
        Positonal function that runs at the beginning for every validation epoch
        """
        ...

    def on_validation_epoch_end(self) -> None:
        r"""
        Positonal function that runs at the end for every training epoch
        """
        ...

    def on_start(self) -> None:
        r"""
        Positonal function that runs at the beginning
        """
        ...

    def on_end(self) -> None:
        r"""
        Positonal function that runs at the end
        """
        ...

    def _training_epoch(self) -> None:
        r"""
        Training epoch function along with callbacks
        """
        self._curr_train_batch_n = 0
        pb = self._create_progressbar()
        for batch in pb:
            self._callback_on_training_batch_start()
            self.on_training_batch_start(pb, batch)
            output = self.training_step(pb, batch)
            self.metrics.cumulate(output, self._stage)
            self.on_training_batch_end(pb, batch, output)
            self._callback_on_training_batch_end()
            self._curr_train_batch_n += 1

    def _validation_epoch(self) -> None:
        r"""
        Validation epoch function along with callbacks
        """
        with torch.no_grad():
            self._curr_val_batch_n = 0
            pb = self._create_progressbar()
            for batch in pb:
                self._callback_on_validation_batch_start()
                self.on_validation_batch_start(pb, batch)
                output = self.validation_step(pb, batch)
                self.metrics.cumulate(output, self._stage)
                self.on_validation_batch_end(pb, batch, output)
                self._callback_on_validation_batch_end()
                self._curr_val_batch_n += 1

    def _train_segment(self) -> None:
        r"""
        Training segment function along with callbacks
        """
        self._callback_on_training_epoch_start()
        self.on_training_epoch_start()
        self._training_epoch()
        self.metrics.normalize(self._stage)
        self.on_training_epoch_end()
        self._callback_on_training_epoch_end()
        self._auto_transfer_scalars("epoch")

    def _val_segment(self) -> None:
        r"""
        Validation segment function along with callbacks
        """
        self._callback_on_validation_epoch_start()
        self.on_validation_epoch_start()
        self._validation_epoch()
        self.metrics.normalize(self._stage)
        self.on_validation_epoch_end()
        self._callback_on_validation_epoch_end()
        self._auto_transfer_scalars("epoch")

    def _print_info(self) -> None:
        r"""
        Prints simple trainer information
        """
        print(f">>> Using device: {str(self._device)}")
        print(
            f">>> Training is enabled"
            if self.training_enabled
            else ">>> Training is disabled"
        )
        print(
            f">>> validation is enabled"
            if self.validation_enabled
            else ">>> Validation is disabled"
        )
        print(f">>> Starting epoch: {self._start_epoch}")
        print(f">>> Ending epoch: {self._end_epoch}")

    def _main(self) -> None:
        r"""
        Main trainer function
        """

        if self._resume:
            with StageContext(self, Stage.LOADING_CP):
                self._load_checkpoints()

        self._print_info()

        self._callback_on_start()
        self.on_start()

        for epoch in range(self._start_epoch, self._end_epoch + 1):
            self._curr_epoch = epoch
            if self.training_enabled:
                with StageContext(self, Stage.TRAINING):
                    self._train_segment()
            if self.validation_enabled:
                with StageContext(self, Stage.VALIDATION):
                    self._val_segment()

            self.schedulers.step_all()

            with StageContext(self, Stage.SAVING_CP):
                if self._force_save_checkpoint:
                    self._save_checkpoints(self._force_save_checkpoint)
                    self._force_save_checkpoint = None
                if self.on_saving_epoch or self.on_last_epoch:
                    self._save_checkpoints()
                if self._terminate:
                    self._save_checkpoints()
                    break

        self.on_end()
        self._callback_on_end()

        print(">>> finishing...")
        self._save_stats()
        if self.tensorboard_enabled:
            self._save_scalars()
            self._save_images()

        if self._zip_at_the_end:
            print(">>> zipping...")
            make_archive(self._output_path, self._output_path)

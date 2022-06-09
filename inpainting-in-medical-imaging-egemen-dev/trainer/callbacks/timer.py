import time

from .base import Callback


class TimerCallback(Callback):
    def __init__(self):
        self.start_time = None
        self.total_train_epoch_time = 0.0
        self.total_train_batch_time = 0.0
        self.total_val_epoch_time = 0.0
        self.total_val_batch_time = 0.0

        self.prev_train_epoch_time = 0.0
        self.prev_train_batch_time = 0.0
        self.prev_val_epoch_time = 0.0
        self.prev_val_batch_time = 0.0

    def on_start(self, trainer) -> None:
        self.start_time = time.time()

    def on_end(self, trainer):
        total_exec_time = self.format_seconds_to_hhmmss(time.time() - self.start_time)
        trainer.stats["total_execution_time"] = total_exec_time
        print(f">>> Total execution time: {total_exec_time} seconds")

        if trainer.training_enabled:
            avg_train_epoch_time = self.total_train_epoch_time / (
                trainer.current_epoch + 1
            )
            trainer.stats[
                "average_training_epoch_execution_time"
            ] = avg_train_epoch_time
            avg_train_batch_time = self.total_train_batch_time / (
                trainer._train_n_batches * (trainer.current_epoch + 1)
            )
            trainer.stats["average_training_step_execution_time"] = avg_train_batch_time
            print(f">>> Avg training epoch time: {avg_train_epoch_time:.3f} seconds")
            print(f">>> Avg training step time: {avg_train_batch_time:.3f} seconds")

        if trainer.validation_enabled:
            avg_val_epoch_time = self.total_val_epoch_time / (trainer.current_epoch + 1)
            trainer.stats[
                "average_validation_epoch_execution_time"
            ] = avg_val_epoch_time
            avg_val_batch_time = self.total_val_batch_time / (
                trainer._val_n_batches * (trainer.current_epoch + 1)
            )
            trainer.stats["average_validation_step_execution_time"] = avg_val_batch_time
            print(f">>> Avg validation epoch time: {avg_val_epoch_time:.3f} seconds")
            print(f">>> Avg validation step time: {avg_val_batch_time:.3f} seconds")

    def on_training_epoch_start(self, trainer):
        if trainer.training_enabled:
            self.prev_train_epoch_time = time.time()

    def on_training_epoch_end(self, trainer):
        if trainer.training_enabled:
            elapsed = time.time() - self.prev_train_epoch_time
            self.total_train_epoch_time += elapsed

    def on_validation_epoch_start(self, trainer):
        if trainer.validation_enabled:
            self.prev_val_epoch_time = time.time()

    def on_validation_epoch_end(self, trainer):
        if trainer.validation_enabled:
            elapsed = time.time() - self.prev_val_epoch_time
            self.total_val_epoch_time += elapsed

    def on_training_batch_start(self, trainer):
        if trainer.training_enabled:
            self.prev_train_batch_time = time.time()

    def on_training_batch_end(self, trainer):
        if trainer.training_enabled:
            elapsed = time.time() - self.prev_train_batch_time
            self.total_train_batch_time += elapsed

    def on_validation_batch_start(self, trainer):
        if trainer.validation_enabled:
            self.prev_val_batch_time = time.time()

    def on_validation_batch_end(self, trainer):
        if trainer.validation_enabled:
            elapsed = time.time() - self.prev_val_batch_time
            self.total_val_batch_time += elapsed

    @staticmethod
    def format_seconds_to_hhmmss(seconds):
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return "%02ih:%02im:%02is" % (hours, minutes, seconds)

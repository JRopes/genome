from trainer.callbacks.base import Callback


class CallbackMixin:
    r"""
    Mixin function for callbacks
    """

    def add_callback(self, callback: Callback) -> None:
        r"""
        Add a callback

        Args:
            callback (Callback): callback to be added
        """
        if isinstance(callback, Callback):
            self.callbacks.append(callback)
        else:
            raise TypeError(f"Expected 'Callback' Instance but got '{type(callback)}'")

    def _callback_on_start(self) -> None:
        r"""
        Runs the `on_start` function of the callbacks
        """
        for cb in self.callbacks:
            cb.on_start(self)

    def _callback_on_end(self) -> None:
        r"""
        Runs the `on_end` function of the callbacks
        """
        for cb in self.callbacks:
            cb.on_end(self)

    def _callback_on_training_batch_start(self) -> None:
        r"""
        Runs the `on_training_batch_start` function of the callbacks
        """
        for cb in self.callbacks:
            cb.on_training_batch_start(self)

    def _callback_on_training_batch_end(self) -> None:
        r"""
        Runs the `on_training_batch_end` function of the callbacks
        """
        for cb in self.callbacks:
            cb.on_training_batch_end(self)

    def _callback_on_validation_batch_start(self) -> None:
        r"""
        Runs the `on_validation_batch_start` function of the callbacks
        """
        for cb in self.callbacks:
            cb.on_validation_batch_start(self)

    def _callback_on_validation_batch_end(self) -> None:
        r"""
        Runs the `on_validation_batch_end` function of the callbacks
        """
        for cb in self.callbacks:
            cb.on_validation_batch_end(self)

    def _callback_on_training_epoch_start(self) -> None:
        r"""
        Runs the `on_training_epoch_start` function of the callbacks
        """
        for cb in self.callbacks:
            cb.on_training_epoch_start(self)

    def _callback_on_training_epoch_end(self) -> None:
        r"""
        Runs the `on_training_epoch_end` function of the callbacks
        """
        for cb in self.callbacks:
            cb.on_training_epoch_end(self)

    def _callback_on_validation_epoch_start(self) -> None:
        r"""
        Runs the `on_validation_epoch_start` function of the callbacks
        """
        for cb in self.callbacks:
            cb.on_validation_epoch_start(self)

    def _callback_on_validation_epoch_end(self) -> None:
        r"""
        Runs the `on_validation_epoch_end` function of the callbacks
        """
        for cb in self.callbacks:
            cb.on_validation_epoch_end(self)

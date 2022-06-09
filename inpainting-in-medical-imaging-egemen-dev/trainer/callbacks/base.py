class Callback:
    def on_start(self, trainer) -> None:
        r"""
        Positonal callback function that runs at the beginning

        Args:
            trainer: trainer instance
        """
        ...

    def on_end(self, trainer) -> None:
        r"""
        Positonal callback function that runs at the end

        Args:
            trainer: trainer instance
        """
        ...

    def on_training_batch_start(self, trainer) -> None:
        r"""
        Positonal callback function that runs at the beginning for every training batch

        Args:
            trainer: trainer instance
        """
        ...

    def on_training_batch_end(self, trainer) -> None:
        r"""
        Positonal callback function that runs at the end for every training batch

        Args:
            trainer: trainer instance
        """
        ...

    def on_validation_batch_start(self, trainer) -> None:
        r"""
        Positonal callback function that runs at the beginning for every validation batch

        Args:
            trainer: trainer instance
        """
        ...

    def on_validation_batch_end(self, trainer) -> None:
        r"""
        Positonal callback function that runs at the end for every training batch

        Args:
            trainer: trainer instance
        """
        ...

    def on_training_epoch_start(self, trainer) -> None:
        r"""
        Positonal callback function that runs at the beginning for every training epoch

        Args:
            trainer: trainer instance
        """
        ...

    def on_training_epoch_end(self, trainer) -> None:
        r"""
        Positonal callback function that runs at the end for every training epoch

        Args:
            trainer: trainer instance
        """
        ...

    def on_validation_epoch_start(self, trainer) -> None:
        r"""
        Positonal callback function that runs at the beginning for every validation epoch

        Args:
            trainer: trainer instance
        """
        ...

    def on_validation_epoch_end(self, trainer) -> None:
        r"""
        Positonal callback function that runs at the end for every training epoch

        Args:
            trainer: trainer instance
        """
        ...

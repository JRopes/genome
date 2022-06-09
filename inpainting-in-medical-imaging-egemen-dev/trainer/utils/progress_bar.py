import sys

from torch.utils.data import DataLoader


class ProgressBar:
    def __init__(
        self,
        dataloader: DataLoader,
        curr_epoch: int,
        pre_desc: str = "",
        post_desc: str = "",
    ):
        r"""
        Very experimental progress bar implementation
        """
        self.it = iter(dataloader)
        self.max = len(dataloader) + 1
        self.curr_epoch = curr_epoch
        self.pre_desc = pre_desc
        self.post_desc = post_desc
        self.postfix = ""
        self.n = 1

    def __iter__(self):
        self.n = 1
        return self

    def __next__(self):
        if self.n <= self.max:
            self._print()
            self.n += 1
            return next(self.it)
        else:
            raise StopIteration

    def _print(self):
        # TODO fix, does not print the last one
        sys.stdout.write("\r")
        ratio = round(100 * (self.n) / (self.max))
        sys.stdout.write(
            f"[{self.curr_epoch}] {self.pre_desc} | {ratio}% - {self.n}/{self.max} | {self.post_desc}{self.postfix}"
        )
        sys.stdout.flush()
        if self.n == self.max:
            sys.stdout.write("\n")

    def set_postfix(self, values: dict) -> None:
        postfix = ""
        for k, v in values.items():
            postfix += f" {k}: {str(v)}"
        self.postfix = postfix

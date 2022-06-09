import os, shutil
from pathlib import Path
from typing import Union

from trainer.typing import PathType


def mkdir(path: Union[str, Path]) -> None:
    r"""
    Create an empty directory if it does not exist

    Args:
        path (str, Path): target directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def make_archive(source: PathType, destination: PathType, format: str = "zip"):
    r"""
    Archive the given folder

    Args:
        source (PathType): path of the source folder
        destination (PathType): path of the destination folder
        format (str): format of the archive file
    """
    base = os.path.basename(destination)
    name = base.split(".")[0]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move("%s.%s" % (name, format), destination)

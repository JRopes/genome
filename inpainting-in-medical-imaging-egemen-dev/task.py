from argparse import ArgumentParser, Namespace
import yaml

from src.utils import get_project_root
from src.preprocess import preprocess
from src.train import train


PROJECT_ROOT_PATH = get_project_root()


def get_args() -> Namespace:
    """Parses given arguments

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser(description="parameters")
    parser.add_argument("--preprocess", default=False, action="store_true")
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--config", type=str, help="path of the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    assert (args.preprocess and not args.train) or (not args.preprocess and args.train)

    with open(args.config) as file:
        config = yaml.safe_load(file)

    if args.preprocess:
        preprocess(PROJECT_ROOT_PATH, config)
    elif args.train:
        train(PROJECT_ROOT_PATH, config)

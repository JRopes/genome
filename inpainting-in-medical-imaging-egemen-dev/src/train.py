import os
import yaml

from src.dataset import BraTSDataset
from torch.utils.data import DataLoader

from trainer import seed_everything

from src.experiments.mask import MaskExperiment
from src.experiments.baseline import BaselineExperiment
from src.experiments.segmentation import SegmentationExperiment


def train(root_path: str, config: dict):
    r"""

    Main function for training/validation

    Args:
        root_path (str): absolute root path
        config (dict): config dictionary

    Necessary elements of config::

        io:
            absolute_paths: bool
            input_path: str
            indices_path: None or str
            output_root_path: str
            resume_path: None or str
        utility:
            random_seed: int
            pin_memory: bool
            num_workers: int
            persistent_workers: bool
            mixed_precision: bool
            pytorch_benchmark: bool
            pytorch_deterministic: bool
            use_tensorbord: bool
            tensorboard_level: "auto" or "manual"
            use_tqdm_progress_bar: bool
            zip_at_the_end: bool
        general:
            training: bool
            validate: bool
            experiment_name: str
            experiment_type: str
            epochs: int
            batch_size: int
            save_epoch_interval: int
        params:
            ...
    """

    absolute_paths = config["io"]["absolute_paths"]
    input_path = config["io"]["input_path"]
    indices_path = config["io"]["indices_path"]
    output_root_path = config["io"]["output_root_path"]
    resume_path = config["io"]["resume_path"]

    random_seed = config["utility"]["random_seed"]
    pin_memory = config["utility"]["pin_memory"]
    num_workers = config["utility"]["num_workers"]
    persistent_workers = config["utility"]["persistent_workers"]
    mixed_precision = config["utility"]["mixed_precision"]
    pytorch_benchmark = config["utility"]["pytorch_benchmark"]
    pytorch_deterministic = config["utility"]["pytorch_deterministic"]
    tensorboard_level = config["utility"]["tensorboard_level"]
    use_tqdm_progress_bar = config["utility"]["use_tqdm_progress_bar"]
    zip_at_the_end = config["utility"]["zip_at_the_end"]

    training = config["general"]["training"]
    validate = config["general"]["validate"]
    experiment_name = config["general"]["experiment_name"]
    experiment_type = config["general"]["experiment_type"]
    epochs = config["general"]["epochs"]
    batch_size = config["general"]["batch_size"]
    save_epoch_interval = config["general"]["save_epoch_interval"]

    params = config["params"]

    c_noise = None
    r_noise = None
    if "noise" in config["params"]:
        if "cocentric_circle_values" in config["params"]["noise"]:
            if config["params"]["noise"]["cocentric_circle_values"] is not None:
                c_noise = config["params"]["noise"]["cocentric_circle_values"]
                c_noise = (c_noise["mean"], c_noise["std"])
        if "cocentric_circle_radiuses" in config["params"]["noise"]:
            if config["params"]["noise"]["cocentric_circle_radiuses"] is not None:
                r_noise = config["params"]["noise"]["cocentric_circle_radiuses"]
                r_noise = (r_noise["mean"], r_noise["std"])

    seed_everything(random_seed)

    if absolute_paths:
        input_path = os.path.join(root_path, input_path)
        output_root_path = os.path.join(root_path, output_root_path)
        if resume_path:
            resume_path = os.path.join(root_path, resume_path)
        if indices_path:
            indices_path = os.path.join(root_path, indices_path)

    if indices_path:
        with open(indices_path) as file:
            indices = yaml.safe_load(file)
    else:
        indices = {"train": None, "val": None}

    if experiment_name:
        output_root_path = os.path.join(output_root_path, experiment_name)

    training_dataset = None
    training_loader = None
    validation_dataset = None
    validation_loader = None

    if training:
        training_dataset = BraTSDataset(
            root=os.path.join(input_path, "train"),
            mode="train",
            fixed_indices=indices["train"],
            add_noise_to_circle_values=c_noise,
            add_noise_to_circle_radius=r_noise,
        )
        training_loader = DataLoader(
            dataset=training_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=persistent_workers,
        )
    if validate:
        validation_dataset = BraTSDataset(
            root=os.path.join(input_path, "validation"),
            mode="validation",
            fixed_indices=indices["val"],
        )

        validation_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=persistent_workers,
        )

    trainer_params = {
        "training_loader": training_loader,
        "validation_loader": validation_loader,
        "epochs": epochs,
        "checkpoint_save_gap": save_epoch_interval,
        "params": params,
        "output_path": output_root_path,
        "resume_path": resume_path,
        "mixed_precision": mixed_precision,
        "benchmark": pytorch_benchmark,
        "deterministic": pytorch_deterministic,
        "tensorboard_level": tensorboard_level,
        "tqdm_progress_bar": use_tqdm_progress_bar,
        "zip_at_the_end": zip_at_the_end,
    }

    if experiment_type == "baseline":
        BaselineExperiment(**trainer_params).start()
    elif experiment_type == "custom_mask":
        MaskExperiment(**trainer_params).start()
    elif experiment_type == "segmentation":
        SegmentationExperiment(**trainer_params).start()
    else:
        raise ValueError(
            f">>> Given experiment_type '{experiment_type}' in the config is not supported!"
        )

# Inpainting in Medical Imaging

---

Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.

## Requirements

* Install [Python](https://www.python.org/downloads/) (tested on 3.8.12).
* Install [PyTorch](https://pytorch.org/get-started/locally/) (tested on 1.10.0).
* Run the command `pip install -r requirements.txt`

## Preprocessing

There are default configuration files for preprocessing but you can modify them or create your own configuration file; however, they need to have specific variables to work with (See [Help](#preprocessing-configuration)).

* For **BraTS20 Dataset**, run the following command with [default BraTS20 configuration](configs/preprocessing/brats20.yml):

```bash
python task.py --preprocess --config="configs/preprocessing/brats20.yml"
```

* For **BraTS18 Dataset**, run the following command with [default BraTS18 configuration](configs/preprocessing/brats18.yml):

```bash
python task.py --preprocess --config="configs/preprocessing/brats18.yml"
```

## Training

Several default configurations are used for training:
* Baseline (Kim et al. 2020)
  * [Full Pipeline](configs/experiments/baseline/full.yml)
  * [Tumor Shape-Grade Segmentation](configs/experiments/baseline/shape_grade.yml)
  * [Tumor Shape Segmentation](configs/experiments/baseline/shape.yml)
  * [Tumor Grade Segmentation](configs/experiments/baseline/grade.yml)
  * [Tumor Inpainting](configs/experiments/baseline/inpainting.yml)
* Our Augmentation Models
  * -
  * -
* Our Inpainting Models
  * -
  * -

You can modify them or create your own configuration file; however, they need to have specific variables to work with (See [Help](#experiment-configuration)).

In order to run such trainings, you must run the following command by only changing the `--config` argument according to your desired configuration path which are located in the `configs/experiments` folder.

Example is as follows:

```bash
python task.py --train --config="configs/experiments/baseline/shape.yml"
```

## Pretrained Models

TO-BE-ADDED

## Augmentation

TO-BE-IMPLEMENTED

```bash
python task.py --augment --config="to_be_implemented"
```

## TensorBoard Support

Visualization via [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) for training and validation is supported. Run the following command to view progress:

```bash
tensorboard --logdir "results" --port 6006 --samples_per_plugin "scalars=200,images=200"
```


## Configuration Help

Configuration system relies on `yaml` files. Make sure to use `42` for `random_seed` in all configurations for reproducibility.

### Preprocessing Configuration

```yaml
io:
  absolute_paths: false                     # indticate whether given paths are absolute or relativ
  input_path: "path_to_raw_dataset"         # path of raw dataset
  output_root_path: "path_to_dump"          # path to dump preprocessed data
utility:
  random_seed: 42                           # seed for randomness
  num_workers: 4                            # number of workers in preprocessing functions
general:
  dataset_type: "brats20"                   # type of the dataset
  validation_split_ratio: 0.1               # ratio of validaton split
params:
  ...                                       # params that are custom to dataset type
```

### Experiment Configuration


```yaml
io:
  absolute_paths: false                     # indticate whether given paths are absolute or relative
  input_path: "path_to_dataset"             # path of preprocessed dataset
  indices_path: "path_to_indices"           # path of indice file
  output_root_path: "path_to_results"       # path to dump results
  resume_path: "path_to_checkpoints"        # path of checkpoints
utility:
  random_seed: 42                           # seed for randomness
  pin_memory: true                          # enable pin memory in data loaders
  num_workers: 4                            # number of workers in data loaders
  persistent_workers: true                  # enable persistent_workers in data loaders
  mixed_precision: false                    # use pytorch mixed precision property (requires manual implementation)
  pytorch_benchmark: true                   # use pytorch benchmark property
  pytorch_deterministic: true               # use pytorch deterministic property
  use_tensorbord: true                      # enable tensorboard
  tensorboard_level: "auto"                 # "auto" or "manual"
  use_tqdm_progress_bar: true               # use tqdm progress bar
  zip_at_the_end: true                      # zip the results at the end
general:
  training: true                            # enable training
  validate: true                            # enable validation
  experiment_name: "inpainting"             # experiment name
  experiment_type: "baseline"               # experiment type
  epochs: 200                               # number of epochs
  batch_size: 32                            # number of batches
  save_epoch_interval: 10                   # epoch gap between checkpoint saves
params:
  ...                                       # params that are custom to experiment type
```

### Augmentation Configuration

```yaml
io:
  ...
utility:
  ...
general:
  ...
params:
  ...
```
# neural-network

![machine-learning](https://img.shields.io/badge/machine--learning-gray)
![neural-network](https://img.shields.io/badge/neural--network-gray)
![python](https://img.shields.io/badge/python-gray)

A [Python](https://python.org) implementation of an artificial neural network (specifically a
multilayer perceptron) with no third-party dependencies, designed to operate on the
[EMNIST dataset](https://nist.gov/itl/products-and-services/emnist-dataset).

## Dataset

The [EMNIST dataset](https://nist.gov/itl/products-and-services/emnist-dataset) should be obtained
from [NIST](https://nist.gov) in
[binary format](https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip) and the archive should be
extracted. The path of the directory containing the extracted contents should be provided to the
programs as necessary. The archives contained within this directory do not need to be extracted.

If the letters dataset is to be used, the correction program (`labels.py`) should be run first.

## Programs

Argument names must be prefixed with `--`.

Default values are provided for all arguments which are not passed to the executed program. Some of
these values may not be suitable for all system configurations (e.g. the dataset may be located in a
different directory), in which case they will need to be overridden.

### [`labels.py`](./source/labels.py)

Corrects a labelling issue in the letters dataset.

#### Arguments

| Name        | Description                                              | Type     | Type Constraints                                 | Example             |
| ----------- | -------------------------------------------------------- | -------- | ------------------------------------------------ | ------------------- |
| `directory` | The path of the directory containing the EMNIST dataset. | `string` | Valid path to a readable and writable directory. | `~/datasets/emnist` |

### [`network.py`](./source/network.py)

Creates and trains the network.

#### Arguments

| Name                             | Description                                                      | Type                                                                                                 | Type Constraints                                                   | Example                |
| -------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ---------------------- |
| `activation-function-identifier` | The identifier of the activation function to use.                | `literal["hyperbolic_tangent", "identity", "recitified_linear_unit", "sigmoid", "softplus", "step"]` |                                                                    | `sigmoid`              |
| `batch-size`                     | The number of samples in each batch.                             | `integer`                                                                                            | `a > 0`                                                            | `10`                   |
| `dataset`                        | The dataset (within the EMNIST dataset) to use.                  | `literal["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]`                            |                                                                    | `digits`               |
| `directory`                      | The path of the directory containing the EMNIST dataset.         | `string`                                                                                             | Valid path to a readable directory.                                | `~/datasets/emnist`    |
| `epoch-count`                    | The number of epochs to train for.                               | `integer`                                                                                            | `a > 0`                                                            | `10`                   |
| `input-model-file`               | The path of the model file to load from.                         | `string`                                                                                             | Valid path to a readable JSON file conforming to the model schema. | `~/models/digits.json` |
| `layer-sizes`                    | The sizes of the layers in the network.                          | `list[integer]`.                                                                                     | Comma-separated, elements satisfy `a > 0`, no parentheses.         | `784,25,10`            |
| `learning-rate`                  | The step size used by the stochastic gradient descent algorithm. | `number`                                                                                             | `a > 0`                                                            | `3`                    |
| `output-model-file`              | The path of the model file to save to.                           | `string`                                                                                             | Valid path to a writable location.                                 | `~/models/digits.json` |
| `sample-count`                   | The number of samples to use from the training split.            | `integer`                                                                                            | `a > 0`                                                            | `240000`               |
| `test-sample-count`              | The number of samples to use from the testing split.             | `integer`                                                                                            | `a > 0`                                                            | `40000`                |

### [`parser.py`](./source/parser.py)

Parses and prints samples from the dataset.

#### Arguments

| Name            | Description                                              | Type                                                                      | Type Constraints                                                              | Example             |
| --------------- | -------------------------------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------- |
| `count`         | The number of samples to load.                           | `integer`                                                                 | `a > 0`                                                                       | `240000`            |
| `dataset`       | The dataset (within the EMNIST dataset) to use.          | `literal["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]` |                                                                               | `digits`            |
| `dataset-split` | The split of the dataset to use.                         | `literal["test", "train"]`                                                |                                                                               | `train`             |
| `directory`     | The path of the directory containing the EMNIST dataset. | `string`                                                                  | Valid path to a readable directory.                                           | `~/datasets/emnist` |
| `output`        | The indices of the samples to output (starting at zero). | `list[string]`                                                            | Comma-separated, elements satisfy `^([0-9]+)(?:-([0-9]+))?$`, no parentheses. | `0,2,4-8`           |

## Tooling

[Prettier](https://prettier.io) is used for formatting of supported non-Python files, and is
configured in [`.prettierrc`](./.prettierrc). \
[Pyrefly](https://pyrefly.org) is used for type checking, and is configured in
[`pyrefly.toml`](./pyrefly.toml). \
[Ruff](https://astral.sh/ruff) is used for formatting and linting of Python files, and is configured
in [`ruff.toml`](./ruff.toml).

These tools must all be installed separately. It is recommended to use the latest versions to avoid
issues and for best results.

All tools should be run from the [project root directory](./).

| Task                  | Command              |
| --------------------- | -------------------- |
| Clean-Up (Ruff)       | `ruff clean`         |
| Formatting (Prettier) | `prettier . --write` |
| Formatting (Ruff)     | `ruff format`        |
| Linting               | `ruff check`         |
| Type Checking         | `pyrefly check`      |

## Usage

**Command:** `python source/<program> [arguments]`\
**Required Python Version:** `>=3.12`\
**Working Directory:** [Project root](./).

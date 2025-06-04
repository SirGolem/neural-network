import enum
import gzip
import os
import pathlib
import struct
import typing

import library.error
import library.interface
import library.module

# Types


class Argument(enum.Enum):
    DATASET = "dataset"
    DATASET_SPLIT = "dataset-split"
    DIRECTORY = "directory"


class Dataset(enum.Enum):
    BALANCED = "balanced"
    BY_CLASS = "byclass"
    BY_MERGE = "bymerge"
    DIGITS = "digits"
    LETTERS = "letters"
    MNIST = "mnist"


class DatasetSplit(enum.Enum):
    TEST = "test"
    TRAIN = "train"


type Arguments = dict[Argument, str]
Defaults = typing.TypedDict(
    "Defaults", {"dataset": Dataset, "dataset_split": DatasetSplit, "directory": pathlib.Path}
)
FileNames = typing.TypedDict(
    "FileNames",
    {
        "images": typing.Callable[[Dataset, DatasetSplit], str],
        "label_mappings": typing.Callable[[Dataset], str],
        "labels": typing.Callable[[Dataset, DatasetSplit], str],
    },
)
type Label = str
type LabelMappings = dict[int, Label]
type ListImage = list[list[int]]
type TupleImage = tuple[tuple[int, ...], ...]


# Constants


DEFAULTS: Defaults = {
    "dataset": Dataset.DIGITS,
    "dataset_split": DatasetSplit.TRAIN,
    "directory": pathlib.Path(os.getcwd()),
}

ESCAPE_CODE_PREFIX: str = "\x1b"
ESCAPE_CODE_RESET: str = f"{ESCAPE_CODE_PREFIX}[0m"

FILE_NAMES: FileNames = {
    "images": lambda dataset,
    dataset_split: f"emnist-{dataset.value}-{dataset_split.value}-images-idx3-ubyte.gz",
    "label_mappings": lambda dataset: f"emnist-{dataset.value}-mapping.txt",
    "labels": lambda dataset,
    dataset_split: f"emnist-{dataset.value}-{dataset_split.value}-labels-idx1-ubyte.gz",
}

MAGIC_NUMBERS: dict[str, int] = {"images": 2051, "labels": 2049}

UNICODE_UPPER_HALF_BLOCK: str = chr(9600)


# Functions


def __check_file_exists(path: pathlib.Path) -> None:
    if not path.exists():
        raise library.error.GracefulError(f"'{path}' does not exist.")
    if not path.is_file():
        raise library.error.GracefulError(f"'{path}' is not a file.")


def __check_magic_number(file: gzip.GzipFile, magic_number: int) -> None:
    file_magic_number: int = struct.unpack(">I", file.read(4))[0]
    if file_magic_number != magic_number:
        raise library.error.GracefulError(
            f"Magic number does not match expected value: Expected '{magic_number}', Found '{file_magic_number}'."
        )


def parse_images(
    dataset: Dataset, dataset_split: DatasetSplit, directory: pathlib.Path
) -> tuple[int, int, int, tuple[TupleImage, ...]]:
    path = pathlib.Path(directory, FILE_NAMES["images"](dataset, dataset_split))
    __check_file_exists(path)

    with gzip.open(path, "rb") as file:
        __check_magic_number(file, MAGIC_NUMBERS["images"])

        image_count: int = struct.unpack(">I", file.read(4))[0]
        image_rows: int = struct.unpack(">I", file.read(4))[0]
        image_columns: int = struct.unpack(">I", file.read(4))[0]
        print(f"Reading {image_count} images ({image_rows}x{image_columns} pixels)...")

        images_list: list[ListImage] = []

        for _ in range(image_count):
            image = [[0 for _ in range(image_columns)] for _ in range(image_rows)]

            for column in range(image_columns):
                for row in range(image_rows):
                    pixel: int = struct.unpack("B", file.read(1))[0]
                    image[row][column] = pixel

            images_list.append(image)

        images_tuple: tuple[TupleImage, ...] = tuple(
            [tuple([tuple(row) for row in image]) for image in images_list]
        )

        print("Read images.")
        return (image_count, image_rows, image_columns, images_tuple)


def parse_label_mappings(dataset: Dataset, directory: pathlib.Path) -> LabelMappings:
    path = pathlib.Path(directory, FILE_NAMES["label_mappings"](dataset))
    __check_file_exists(path)

    with open(path, "rt") as file:
        print("Reading label mappings...")

        mappings: LabelMappings = {}

        for line in file.readlines():
            line = line.strip().split(" ")
            raw_value = int(line[0])
            character = chr(int(line[1]))
            mappings[raw_value] = character

        print("Read label mappings.")
        return mappings


def parse_labels(
    dataset: Dataset, dataset_split: DatasetSplit, directory: pathlib.Path, mappings: LabelMappings
) -> tuple[int, tuple[Label, ...]]:
    path = pathlib.Path(directory, FILE_NAMES["labels"](dataset, dataset_split))
    __check_file_exists(path)

    with gzip.open(path, "rb") as file:
        __check_magic_number(file, MAGIC_NUMBERS["labels"])

        label_count: int = struct.unpack(">I", file.read(4))[0]
        print(f"Reading {label_count} labels...")

        labels: list[Label] = []

        for _ in range(label_count):
            raw_label: int = struct.unpack("B", file.read(1))[0]
            labels.append(mappings[raw_label])

        print("Read labels.")
        return (label_count, tuple(labels))


def print_image(image: TupleImage, label: Label, rows: int, columns: int) -> None:
    header = f"Label: {label} "
    print(header + "-" * (columns - len(header)))

    for row in range(0, rows // 2):
        for column in range(columns):
            upper_half = image[2 * row][column]
            lower_half = image[2 * row + 1][column]
            upper_half_modifier = (
                f"{ESCAPE_CODE_PREFIX}[38;2;{upper_half};{upper_half};{upper_half}m"
            )
            lower_half_modifier = (
                f"{ESCAPE_CODE_PREFIX}[48;2;{lower_half};{lower_half};{lower_half}m"
            )
            print(
                upper_half_modifier
                + lower_half_modifier
                + UNICODE_UPPER_HALF_BLOCK
                + ESCAPE_CODE_RESET,
                end="",
            )

        print()

    print("-" * columns)


# Main Program


def main() -> bool:
    arguments = library.interface.parse_arguments()

    try:
        dataset: Dataset = (
            Dataset(arguments[Argument.DATASET.value])
            if Argument.DATASET.value in arguments
            else DEFAULTS["dataset"]
        )
    except ValueError:
        raise library.error.InvalidArgumentValueError(
            Argument.DATASET.value,
            f"expected one of {', '.join([f"'{value.value}'" for value in Dataset._member_map_.values()])}",
            arguments[Argument.DATASET.value],
        )

    try:
        dataset_split: DatasetSplit = (
            DatasetSplit(arguments[Argument.DATASET_SPLIT.value])
            if Argument.DATASET_SPLIT.value in arguments
            else DEFAULTS["dataset_split"]
        )
    except ValueError:
        raise library.error.InvalidArgumentValueError(
            Argument.DATASET_SPLIT.value,
            f"expected one of {', '.join([f"'{value.value}'" for value in DatasetSplit._member_map_.values()])}",
            arguments[Argument.DATASET_SPLIT.value],
        )

    directory: pathlib.Path = (
        pathlib.Path(arguments[Argument.DIRECTORY.value])
        if Argument.DIRECTORY.value in arguments
        else DEFAULTS["directory"]
    ).expanduser()

    if not directory.exists():
        raise library.error.InvalidArgumentValueError(
            Argument.DIRECTORY.value, "path does not exist", arguments[Argument.DIRECTORY.value]
        )
    if not directory.is_dir():
        raise library.error.InvalidArgumentValueError(
            Argument.DIRECTORY.value, "path is not a directory", arguments[Argument.DIRECTORY.value]
        )

    mappings = parse_label_mappings(dataset, directory)
    label_count, labels = parse_labels(dataset, dataset_split, directory, mappings)
    image_count, image_rows, image_columns, images = parse_images(dataset, dataset_split, directory)

    if label_count != image_count:
        raise library.error.GracefulError(
            f"Label count ({label_count}) does not match image count ({image_count})."
        )

    for i in range(5):
        print_image(images[i], labels[i], image_rows, image_columns)

    return True


if __name__ == "__main__":
    library.module.try_main(main)

import enum
import gzip
import itertools
import os
import pathlib
import re
import struct
import typing

import library.error
import library.interface
import library.logging
import library.matrix
import library.module
import library.type

# Types


class Argument(enum.Enum):
    COUNT = "count"
    DATASET = "dataset"
    DATASET_SPLIT = "dataset-split"
    DIRECTORY = "directory"
    OUTPUT = "output"


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


Defaults = typing.TypedDict(
    "Defaults",
    {
        "count": None | int,
        "dataset": Dataset,
        "dataset_split": DatasetSplit,
        "directory": pathlib.Path,
    },
)

FileNames = typing.TypedDict(
    "FileNames",
    {
        "images": typing.Callable[[Dataset, DatasetSplit], str],
        "label_mappings": typing.Callable[[Dataset], str],
        "labels": typing.Callable[[Dataset, DatasetSplit], str],
    },
)


# Constants


DEFAULTS: Defaults = {
    "count": None,
    "dataset": Dataset.DIGITS,
    "dataset_split": DatasetSplit.TRAIN,
    "directory": pathlib.Path(os.getcwd()),
}

FILE_NAMES: FileNames = {
    "images": (
        lambda dataset,
        dataset_split: f"emnist-{dataset.value}-{dataset_split.value}-images-idx3-ubyte.gz"
    ),
    "label_mappings": lambda dataset: f"emnist-{dataset.value}-mapping.txt",
    "labels": (
        lambda dataset,
        dataset_split: f"emnist-{dataset.value}-{dataset_split.value}-labels-idx1-ubyte.gz"
    ),
}

MAGIC_NUMBERS: dict[str, int] = {"images": 2051, "labels": 2049}

MAXIMUM_PIXEL_VALUE: int = 255

OUTPUT_RANGE_REGULAR_EXPRESSION: str = r"^([0-9]+)(?:-([0-9]+))?$"

UNICODE_UPPER_HALF_BLOCK: str = chr(9600)


# Functions


def _check_file_exists(path: pathlib.Path) -> None:
    if not path.exists():
        raise library.error.FileSystemResourceNotFoundError(path)
    if not path.is_file():
        raise library.error.IncorrectFileSystemResourceTypeError("file", path)


def _check_magic_number(file: gzip.GzipFile, magic_number: int) -> None:
    file_magic_number: int = struct.unpack(">I", file.read(4))[0]
    if file_magic_number != magic_number:
        raise library.error.MagicNumberValidationError(magic_number, file_magic_number)


def convert_image_to_matrix(image: library.type.TupleImage) -> library.matrix.Matrix:
    return library.matrix.Matrix.from_tuples(tuple([flatten_image(image)]))


def convert_images_to_matrices(
    images: tuple[library.type.TupleImage, ...],
) -> tuple[library.matrix.Matrix, ...]:
    return tuple(convert_image_to_matrix(image) for image in images)


def convert_label_to_matrix(label: library.type.Label) -> library.matrix.Matrix:
    matrix = library.matrix.Matrix(1, 10)
    matrix.set(0, int(label), 1)
    return matrix


def convert_labels_to_matrices(
    labels: tuple[library.type.Label, ...],
) -> tuple[library.matrix.Matrix, ...]:
    return tuple(convert_label_to_matrix(label) for label in labels)


def flatten_image(image: library.type.TupleImage) -> tuple[float, ...]:
    return tuple(itertools.chain.from_iterable(image))


def flatten_images(images: tuple[library.type.TupleImage, ...]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(itertools.chain.from_iterable(image)) for image in images)


def parse_images(
    dataset: Dataset, dataset_split: DatasetSplit, directory: pathlib.Path, count: None | int = None
) -> tuple[int, int, int, tuple[library.type.TupleImage, ...]]:
    path = pathlib.Path(directory, FILE_NAMES["images"](dataset, dataset_split))
    _check_file_exists(path)

    with gzip.open(path, "rb") as file:
        _check_magic_number(file, MAGIC_NUMBERS["images"])

        image_count: int = struct.unpack(">I", file.read(4))[0]
        if count is not None:
            image_count = min(count, image_count)
        image_rows: int = struct.unpack(">I", file.read(4))[0]
        image_columns: int = struct.unpack(">I", file.read(4))[0]
        progress = library.interface.Progress(
            f"Reading {image_count} images ({image_rows}x{image_columns} pixels): ",
            " (Starting...)",
        )
        progress.print(0)

        images_list: list[library.type.ListImage] = []

        for image_index in range(image_count):
            progress.suffix = f" (Image {image_index + 1}/{image_count})"
            progress.print(image_index / image_count)

            image = [[0.0 for _ in range(image_columns)] for _ in range(image_rows)]

            for column_index in range(image_columns):
                for row_index in range(image_rows):
                    pixel: int = struct.unpack("B", file.read(1))[0]
                    image[column_index][row_index] = pixel / MAXIMUM_PIXEL_VALUE

            images_list.append(image)

        images_tuple: tuple[library.type.TupleImage, ...] = tuple(
            [tuple([tuple(column) for column in image]) for image in images_list]
        )

        progress.complete(f"Read {image_count} images.")
        return (image_count, image_rows, image_columns, images_tuple)


def parse_label_mappings(dataset: Dataset, directory: pathlib.Path) -> library.type.LabelMappings:
    path = pathlib.Path(directory, FILE_NAMES["label_mappings"](dataset))
    _check_file_exists(path)

    with open(path, "rt") as file:
        print("Reading label mappings...", end="")

        mappings: library.type.LabelMappings = {}

        for line in file.readlines():
            split_line = line.strip().split(" ")
            raw_value = int(split_line[0])
            character = chr(int(split_line[1]))
            mappings[raw_value] = character

        library.interface.clear_line()
        print("Read label mappings.")
        return mappings


def parse_labels(
    dataset: Dataset,
    dataset_split: DatasetSplit,
    directory: pathlib.Path,
    mappings: library.type.LabelMappings,
    count: None | int = None,
) -> tuple[int, tuple[library.type.Label, ...]]:
    path = pathlib.Path(directory, FILE_NAMES["labels"](dataset, dataset_split))
    _check_file_exists(path)

    with gzip.open(path, "rb") as file:
        _check_magic_number(file, MAGIC_NUMBERS["labels"])

        label_count: int = struct.unpack(">I", file.read(4))[0]
        if count is not None:
            label_count = min(count, label_count)
        progress = library.interface.Progress(f"Reading {label_count} labels: ", " (Starting...)")
        progress.print(0)

        labels: list[library.type.Label] = []

        for label_index in range(label_count):
            progress.suffix = f" (Label {label_index + 1}/{label_count})"
            progress.print(label_index / label_count)

            raw_label: int = struct.unpack("B", file.read(1))[0]
            labels.append(mappings[raw_label])

        progress.complete(f"Read {label_count} labels.")
        return (label_count, tuple(labels))


def print_image(
    image: library.type.TupleImage, label: library.type.Label, rows: int, columns: int
) -> None:
    header = f"Label: {label} "
    print(header + "-" * (columns - len(header)))

    for row_index in range(0, rows // 2):
        for column_index in range(columns):
            upper_half = int(image[column_index][2 * row_index] * MAXIMUM_PIXEL_VALUE)
            lower_half = int(image[column_index][2 * row_index + 1] * MAXIMUM_PIXEL_VALUE)
            upper_half_modifier = (
                f"{library.logging.ESCAPE_CODE_PREFIX}[38;2;{upper_half};{upper_half};{upper_half}m"
            )
            lower_half_modifier = (
                f"{library.logging.ESCAPE_CODE_PREFIX}[48;2;{lower_half};{lower_half};{lower_half}m"
            )
            print(
                upper_half_modifier
                + lower_half_modifier
                + UNICODE_UPPER_HALF_BLOCK
                + library.logging.ESCAPE_CODE_RESET,
                end="",
            )

        print()

    print("-" * columns)


# Main Program


def main() -> bool:
    arguments = library.interface.parse_arguments()

    count_option = arguments.get_option(Argument.COUNT.value, str(DEFAULTS["count"]))

    try:
        if count_option == str(None):
            count = None
        else:
            count = int(count_option)
            if count < 1:
                raise ValueError
    except ValueError:
        raise library.error.InvalidArgumentValueError(
            Argument.COUNT.value, "expected an integer greater than zero", count_option
        )

    dataset_option = arguments.get_option(Argument.DATASET.value, DEFAULTS["dataset"].value)

    try:
        dataset = Dataset(dataset_option)
    except ValueError:
        raise library.error.InvalidArgumentValueError(
            Argument.DATASET.value,
            f"expected one of {', '.join([f"'{value.value}'" for value in Dataset._member_map_.values()])}",
            dataset_option,
        )

    dataset_split_option = arguments.get_option(
        Argument.DATASET_SPLIT.value, DEFAULTS["dataset_split"].value
    )

    try:
        dataset_split = DatasetSplit(dataset_split_option)
    except ValueError:
        raise library.error.InvalidArgumentValueError(
            Argument.DATASET_SPLIT.value,
            f"expected one of {', '.join([f"'{value.value}'" for value in DatasetSplit._member_map_.values()])}",
            dataset_split_option,
        )

    directory_option = arguments.get_option(Argument.DIRECTORY.value, str(DEFAULTS["directory"]))
    directory = (pathlib.Path(directory_option)).expanduser()

    if not directory.exists():
        raise library.error.InvalidArgumentValueError(
            Argument.DIRECTORY.value, "path does not exist", directory_option
        )

    if not directory.is_dir():
        raise library.error.InvalidArgumentValueError(
            Argument.DIRECTORY.value, "path is not a directory", directory_option
        )

    output: list[int] = []
    output_option = arguments.get_option(Argument.OUTPUT.value, "")
    output_ranges = [
        output_range.strip()
        for output_range in output_option.split(",")
        if len(output_range.strip()) != 0
    ]

    for output_range in output_ranges:
        match_result = re.match(OUTPUT_RANGE_REGULAR_EXPRESSION, output_range)

        try:
            if match_result is None:
                raise ValueError

            range_start = match_result.group(1)
            range_stop = (
                match_result.group(2)
                if match_result.group(2) is not None
                else match_result.group(1)
            )

            for image_index in range(int(range_start), int(range_stop) + 1):
                output.append(image_index)
        except ValueError:
            raise library.error.InvalidArgumentValueError(
                Argument.OUTPUT.value,
                "expected a comma-separated list of positive integers or integer ranges (in the form 'a-b', inclusive)",
                output_option,
            )

    image_count, image_rows, image_columns, images = parse_images(
        dataset, dataset_split, directory, count
    )
    mappings = parse_label_mappings(dataset, directory)
    label_count, labels = parse_labels(dataset, dataset_split, directory, mappings, count)

    if image_count != label_count:
        raise library.error.ImageCountDoesNotMatchLabelCountError(image_count, label_count)

    for image_index in output:
        if image_index >= image_count:
            raise library.error.InvalidArgumentValueError(
                Argument.OUTPUT.value,
                "expected all values (after expansion) to be in the range '0 <= x < [image count]'",
                output_option,
            )

        print_image(images[image_index], labels[image_index], image_rows, image_columns)

    return True


if __name__ == "__main__":
    library.module.try_main(main)

import enum
import gzip
import os
import pathlib
import re
import struct
import typing

import library.error
import library.interface
import library.logging
import library.module

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

type Label = str
type LabelMappings = dict[int, Label]

type ListImage = list[list[float]]
type TupleImage = tuple[tuple[float, ...], ...]


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


def __check_file_exists(path: pathlib.Path) -> None:
    if not path.exists():
        raise library.error.FileSystemResourceNotFoundError(path)
    if not path.is_file():
        raise library.error.IncorrectFileSystemResourceTypeError("file", path)


def __check_magic_number(file: gzip.GzipFile, magic_number: int) -> None:
    file_magic_number: int = struct.unpack(">I", file.read(4))[0]
    if file_magic_number != magic_number:
        raise library.error.MagicNumberValidationError(magic_number, file_magic_number)


def parse_images(
    dataset: Dataset, dataset_split: DatasetSplit, directory: pathlib.Path, count: None | int = None
) -> tuple[int, int, int, tuple[TupleImage, ...]]:
    path = pathlib.Path(directory, FILE_NAMES["images"](dataset, dataset_split))
    __check_file_exists(path)

    try:
        with gzip.open(path, "rb") as file:
            __check_magic_number(file, MAGIC_NUMBERS["images"])

            image_count: int = struct.unpack(">I", file.read(4))[0]
            if count is not None:
                image_count = min(count, image_count)
            image_rows: int = struct.unpack(">I", file.read(4))[0]
            image_columns: int = struct.unpack(">I", file.read(4))[0]
            print(f"Reading {image_count} images ({image_rows}x{image_columns} pixels)...")

            images_list: list[ListImage] = []

            for _ in range(image_count):
                image = [[0.0 for _ in range(image_columns)] for _ in range(image_rows)]

                for column in range(image_columns):
                    for row in range(image_rows):
                        pixel: int = struct.unpack("B", file.read(1))[0]
                        image[row][column] = pixel / MAXIMUM_PIXEL_VALUE

                images_list.append(image)

            images_tuple: tuple[TupleImage, ...] = tuple(
                [tuple([tuple(row) for row in image]) for image in images_list]
            )

            print("Read images.")
            return (image_count, image_rows, image_columns, images_tuple)
    except library.error.ApplicationError as error:
        raise error
    except Exception as error:
        raise library.error.ImagesFileReadError(error)


def parse_label_mappings(dataset: Dataset, directory: pathlib.Path) -> LabelMappings:
    path = pathlib.Path(directory, FILE_NAMES["label_mappings"](dataset))
    __check_file_exists(path)

    try:
        with open(path, "rt") as file:
            print("Reading label mappings...")

            mappings: LabelMappings = {}

            for line in file.readlines():
                split_line = line.strip().split(" ")
                raw_value = int(split_line[0])
                character = chr(int(split_line[1]))
                mappings[raw_value] = character

            print("Read label mappings.")
            return mappings
    except Exception as error:
        raise library.error.LabelMappingsFileReadError(error)


def parse_labels(
    dataset: Dataset,
    dataset_split: DatasetSplit,
    directory: pathlib.Path,
    mappings: LabelMappings,
    count: None | int = None,
) -> tuple[int, tuple[Label, ...]]:
    path = pathlib.Path(directory, FILE_NAMES["labels"](dataset, dataset_split))
    __check_file_exists(path)

    try:
        with gzip.open(path, "rb") as file:
            __check_magic_number(file, MAGIC_NUMBERS["labels"])

            label_count: int = struct.unpack(">I", file.read(4))[0]
            if count is not None:
                label_count = min(count, label_count)
            print(f"Reading {label_count} labels...")

            labels: list[Label] = []

            for _ in range(label_count):
                raw_label: int = struct.unpack("B", file.read(1))[0]
                labels.append(mappings[raw_label])

            print("Read labels.")
            return (label_count, tuple(labels))
    except library.error.ApplicationError as error:
        raise error
    except Exception as error:
        raise library.error.LabelsFileReadError(error)


def print_image(image: TupleImage, label: Label, rows: int, columns: int) -> None:
    header = f"Label: {label} "
    print(header + "-" * (columns - len(header)))

    for row in range(0, rows // 2):
        for column in range(columns):
            upper_half = int(image[2 * row][column] * MAXIMUM_PIXEL_VALUE)
            lower_half = int(image[2 * row + 1][column] * MAXIMUM_PIXEL_VALUE)
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
        if count_option == "None":
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

            for index in range(int(range_start), int(range_stop) + 1):
                output.append(index)
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

    for index in output:
        if index >= image_count:
            raise library.error.InvalidArgumentValueError(
                Argument.OUTPUT.value,
                "expected all values (after expansion) to be in the range '0 <= x < [image count]'",
                output_option,
            )

        print_image(images[index], labels[index], image_rows, image_columns)

    return True


if __name__ == "__main__":
    library.module.try_main(main)

import enum
import gzip
import os
import pathlib
import struct
import typing

import library.error
import library.file_system
import library.interface
import library.module
import library.type
import parser

# Types


class Argument(enum.Enum):
    DIRECTORY = "directory"


Defaults = typing.TypedDict("Defaults", {"directory": pathlib.Path})


# Constants


DATASET: library.type.Dataset = library.type.Dataset.LETTERS
DEFAULTS: Defaults = {"directory": pathlib.Path(os.getcwd())}


# Main Program


def main() -> bool:
    arguments = library.interface.parse_arguments()

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

    label_mappings_file_path = directory.joinpath(parser.FILE_NAMES["label_mappings"](DATASET))
    library.file_system.check_file_exists(label_mappings_file_path)

    label_mappings: list[str] = []

    try:
        with open(label_mappings_file_path, "rt") as file:
            for line in file.readlines():
                split_line = line.strip().split(" ")
                if int(split_line[0]) < 1:
                    raise library.error.CorrectedLabelWouldBeInvalidError
                split_line[0] = str(int(split_line[0]) - 1)
                label_mappings.append(" ".join(split_line) + "\n")
    except library.error.ApplicationError as error:
        raise error
    except Exception as error:
        raise library.error.LabelMappingsFileCorrectionError(error)

    try:
        with open(label_mappings_file_path, "wt") as file:
            file.writelines(label_mappings)
    except Exception as error:
        raise library.error.CorrectedLabelMappingsFileWriteError(error)

    for dataset_split in library.type.DatasetSplit:
        labels_file_path = directory.joinpath(parser.FILE_NAMES["labels"](DATASET, dataset_split))
        library.file_system.check_file_exists(labels_file_path)

        labels = bytes()

        try:
            with gzip.open(labels_file_path, "rb") as file:
                magic_number = parser.MAGIC_NUMBERS["labels"]
                parser.check_magic_number(file, magic_number)
                labels += struct.pack(">I", magic_number)

                label_count: int = struct.unpack(">I", file.read(4))[0]
                labels += struct.pack(">I", label_count)

                for label_index in range(label_count):
                    label: int = struct.unpack("B", file.read(1))[0]
                    labels += struct.pack("B", label - 1)
        except Exception as error:
            raise library.error.LabelsFileCorrectionError(error)

        try:
            with gzip.open(labels_file_path, "wb") as file:
                file.write(labels)
        except Exception as error:
            raise library.error.CorrectedLabelsFileWriteError(error)

    return True


if __name__ == "__main__":
    library.module.try_main(main)

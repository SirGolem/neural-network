import pathlib

import library.error

# Functions


def check_file_exists(path: pathlib.Path) -> None:
    if not path.exists():
        raise library.error.FileSystemResourceNotFoundError(path)
    if not path.is_file():
        raise library.error.IncorrectFileSystemResourceTypeError("file", path)

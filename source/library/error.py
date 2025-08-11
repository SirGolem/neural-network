import pathlib
import typing

import library.logging

# Types


type ArgumentType = typing.Literal["flag"] | typing.Literal["option"]
type FileSystemResourceType = typing.Literal["directory"] | typing.Literal["file"]


# Error Classes


class ApplicationError(Exception):
    def __init__(self: typing.Self, message: str) -> None:
        self.message = message

    def __str__(self: typing.Self) -> str:
        return self.message


class CostCalculationError(ApplicationError):
    def __init__(self: typing.Self, error: Exception) -> None:
        self.message = f"Failed to calculate network cost: {library.logging.error_message(error)}"


class FileSystemResourceNotFoundError(ApplicationError):
    def __init__(self: typing.Self, path: pathlib.Path | str) -> None:
        self.message = f"'{str(path)}' does not exist."


class ImageCountDoesNotMatchLabelCountError(ApplicationError):
    def __init__(self: typing.Self, image_count: int, label_count: int) -> None:
        self.message = f"Image count ({image_count}) does not match label count ({label_count})."


class IncorrectArgumentTypeError(ApplicationError):
    def __init__(self: typing.Self, argument: str, correct_type: ArgumentType) -> None:
        self.message = f"Type of argument '{argument}' is incorrect: expected {'a flag' if correct_type == 'flag' else 'an option'}, received {'an option' if correct_type == 'flag' else 'a flag'}."


class IncorrectFileSystemResourceTypeError(ApplicationError):
    def __init__(
        self: typing.Self, correct_type: FileSystemResourceType, path: pathlib.Path | str
    ) -> None:
        self.message = f"'{str(path)}' is not a {correct_type}."


class IncorrectLayerInputCountError(ApplicationError):
    def __init__(self: typing.Self, expected: int, received: int) -> None:
        self.message = (
            f"Incorrect number of layer inputs: expected '{expected}', received '{received}'."
        )


class InvalidArgumentValueError(ApplicationError):
    def __init__(self: typing.Self, argument: str, reason: str, value: str) -> None:
        self.message = f"Invalid value '{value}' provided for argument '{argument}': {reason}."


class InvalidLayerInputCountError(ApplicationError):
    def __init__(self: typing.Self, received: int) -> None:
        self.message = (
            f"Invalid number of layer inputs: expected at least one, received '{received}'."
        )


class InvalidLayerOutputCountError(ApplicationError):
    def __init__(self: typing.Self, received: int) -> None:
        self.message = (
            f"Invalid number of layer outputs: expected at least one, received '{received}'."
        )


class InvalidNetworkLayerCountError(ApplicationError):
    def __init__(self: typing.Self, received: int) -> None:
        self.message = (
            f"Invalid number of network layers: expected at least one, received '{received}'."
        )


class MagicNumberValidationError(ApplicationError):
    def __init__(self: typing.Self, expected: int, found: int) -> None:
        self.message = f"Magic number does not match expected value: expected '{str(expected)}', found '{str(found)}'."


class MissingRequiredArgumentError(ApplicationError):
    def __init__(self: typing.Self, argument: str) -> None:
        self.message = f"Missing required argument '{argument}'."

import re
import sys
import typing

import library.error

# Classes


class Arguments:
    def __init__(self: typing.Self, arguments: dict[str, None | str]) -> None:
        self.arguments = arguments

    def get_flag(self: typing.Self, argument: str) -> bool:
        if argument in self.arguments and self.arguments[argument] is not None:
            raise library.error.IncorrectArgumentTypeError(argument, "flag")

        return argument in self.arguments

    def get_option(self: typing.Self, argument: str, default: None | str = None) -> str:
        if argument not in self.arguments and default is None:
            raise library.error.MissingRequiredArgumentError(argument)
        if argument in self.arguments and self.arguments[argument] is None:
            raise library.error.IncorrectArgumentTypeError(argument, "option")

        return str(default) if argument not in self.arguments else str(self.arguments[argument])


# Constants


ARGUMENT_REGULAR_EXPRESSION = r"--([a-zA-Z-]+)(?:=(.+))?"


# Functions


def parse_arguments() -> Arguments:
    arguments: dict[str, None | str] = {}

    for argument in sys.argv[1:]:
        match_result = re.fullmatch(ARGUMENT_REGULAR_EXPRESSION, argument)

        if match_result is not None:
            arguments[match_result.group(1)] = match_result.group(2)

    return Arguments(arguments)

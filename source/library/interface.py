import os
import re
import sys
import typing

import library.error

# Constants


ARGUMENT_REGULAR_EXPRESSION = r"--([a-zA-Z-]+)(?:=(.+))?"
PROGRESS_BAR_CHARACTER: str = chr(9608)


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


class Progress:
    def __init__(self: typing.Self, prefix: None | str = None, suffix: None | str = None) -> None:
        self.prefix = prefix
        self.suffix = suffix

    def complete(self: typing.Self, string: None | str = None) -> None:
        if string is None:
            self.print(1)
            print()
        else:
            clear_line()
            print(string)

    def print(self: typing.Self, progress: float) -> None:
        if progress < 0 or progress > 1:
            raise library.error.InvalidProgressValueError(progress)

        terminal_width = os.get_terminal_size().columns
        progress_bar_width = terminal_width - len(self.prefix or ()) - len(self.suffix or ()) - 2
        progress_bar = f"[{(PROGRESS_BAR_CHARACTER * round(progress * progress_bar_width)).ljust(progress_bar_width)}]"
        print(f"\r{self.prefix or ''}{progress_bar}{self.suffix or ''}", end="", flush=True)


# Functions


def clear_line() -> None:
    print(f"\r{' ' * os.get_terminal_size().columns}\r", end="", flush=True)


def parse_arguments() -> Arguments:
    arguments: dict[str, None | str] = {}

    for argument in sys.argv[1:]:
        match_result = re.fullmatch(ARGUMENT_REGULAR_EXPRESSION, argument)

        if match_result is not None:
            arguments[match_result.group(1)] = match_result.group(2)

    return Arguments(arguments)

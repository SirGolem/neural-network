import sys
import typing

# Constants


ESCAPE_CODE_PREFIX: str = "\x1b"
ESCAPE_CODE_RESET: str = f"{ESCAPE_CODE_PREFIX}[0m"


# Classes


class Logger:
    def __init__(self: typing.Self) -> None:
        self.last_character: None | str = None
        self.standard_output = sys.stdout

    def flush(self: typing.Self) -> None:
        self.standard_output.flush()

    def ensure_empty_line(self: typing.Self) -> None:
        if self.last_character is not None and self.last_character != "\n":
            print()

    def write(self: typing.Self, string: str) -> int:
        if len(string) > 0:
            self.last_character = string[-1]

        return self.standard_output.write(string)


# Functions


def error_message(error: Exception) -> str:
    return str(error) or "No message provided."


def print_error(error: Exception) -> None:
    print_error_message(error_message(error))


def print_error_message(message: str) -> None:
    print("Error: " + message, file=sys.stderr)

import re
import sys

# Constants


ARGUMENT_REGULAR_EXPRESSION = r"--([a-zA-Z-]+)(?:=(.+))?"


# Functions


def parse_arguments() -> dict[str, None | str]:
    arguments: dict[str, None | str] = {}

    for argument in sys.argv[1:]:
        match_result = re.fullmatch(ARGUMENT_REGULAR_EXPRESSION, argument)

        if match_result is not None:
            arguments[match_result.group(1)] = match_result.group(2)

    return arguments

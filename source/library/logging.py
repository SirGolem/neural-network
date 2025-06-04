import sys

# Functions


def print_error(message: str) -> None:
    print("Error: " + message, file=sys.stderr)

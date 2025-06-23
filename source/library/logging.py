import sys

# Constants


ESCAPE_CODE_PREFIX: str = "\x1b"
ESCAPE_CODE_RESET: str = f"{ESCAPE_CODE_PREFIX}[0m"


# Functions


def error_message(error: Exception) -> str:
    return str(error) or "No message provided."


def print_error(error: Exception) -> None:
    print_error_message(error_message(error))


def print_error_message(message: str) -> None:
    print("Error: " + message, file=sys.stderr)

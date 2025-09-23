import sys
import traceback
import typing

import library.error
import library.logging

# Constants


EXIT_CODE_FAILURE: int = 1
EXIT_CODE_KEYBOARD_INTERRUPT: int = 130
EXIT_CODE_SUCCESS: int = 0


# Functions


def try_main(main: typing.Callable[[], bool]) -> None:
    sys.stdout = library.logging.Logger()

    try:
        sys.exit(EXIT_CODE_SUCCESS if main() else EXIT_CODE_FAILURE)
    except KeyboardInterrupt:
        print()
        sys.exit(EXIT_CODE_KEYBOARD_INTERRUPT)
    except library.error.ApplicationError as error:
        sys.stdout.ensure_empty_line()
        library.logging.print_error(error)
        sys.exit(EXIT_CODE_FAILURE)
    except Exception as error:
        sys.stdout.ensure_empty_line()
        library.logging.print_error_message(
            f"An unexpected error occurred: {library.logging.error_message(error)}"
        )
        _, _, error_traceback = sys.exc_info()
        traceback.print_tb(error_traceback)
        sys.exit(EXIT_CODE_FAILURE)

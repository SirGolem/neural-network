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
    try:
        sys.exit(EXIT_CODE_SUCCESS if main() else EXIT_CODE_FAILURE)
    except KeyboardInterrupt:
        print()
        sys.exit(EXIT_CODE_KEYBOARD_INTERRUPT)
    except library.error.GracefulError as error:
        library.logging.print_error(str(error) or "No message provided.")
        sys.exit(EXIT_CODE_FAILURE)
    except Exception as error:
        library.logging.print_error(
            f"An unexpected error occurred: {str(error) or 'No message provided.'}"
        )
        _, _, error_traceback = sys.exc_info()
        traceback.print_tb(error_traceback)
        sys.exit(EXIT_CODE_FAILURE)

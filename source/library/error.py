import typing

# Error Classes


class GracefulError(Exception):
    pass


class InvalidArgumentValueError(GracefulError):
    def __init__(self: typing.Self, argument: str, reason: str, value: str) -> None:
        self.message = f"Invalid value '{value}' provided for argument '{argument}': {reason}."

    def __str__(self: typing.Self) -> str:
        return self.message

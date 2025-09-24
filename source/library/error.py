import pathlib
import typing

# Types


type ArgumentType = typing.Literal["flag"] | typing.Literal["option"]
type FileSystemResourceType = typing.Literal["directory"] | typing.Literal["file"]


# Error Classes


class ApplicationError(Exception):
    def __init__(self: typing.Self, message: str) -> None:
        self.message = message

    def __str__(self: typing.Self) -> str:
        return self.message


class LibraryError(Exception):
    def __init__(self: typing.Self, message: str) -> None:
        self.message = message

    def __str__(self: typing.Self) -> str:
        return self.message


class ExpectedResultCountDoesNotMatchSampleCountError(ApplicationError):
    def __init__(self: typing.Self, expected_result_count: int, sample_count: int) -> None:
        self.message = f"Expected result count ({expected_result_count}) does not match sample count ({sample_count})."


class FileSystemResourceNotFoundError(ApplicationError):
    def __init__(self: typing.Self, path: pathlib.Path | str) -> None:
        self.message = f"'{str(path)}' does not exist."


class ImageCountDoesNotMatchLabelCountError(ApplicationError):
    def __init__(self: typing.Self, image_count: int, label_count: int) -> None:
        self.message = f"Image count ({image_count}) does not match label count ({label_count})."


class IncompatibleMatrixDimensionsError(LibraryError):
    def __init__(
        self: typing.Self, a_columns: int, a_rows: int, b_columns: int, b_rows: int
    ) -> None:
        self.message = f"Incompatible matrix dimensions: cannot operate on matrices with dimensions '{a_rows}x{a_columns}' and '{b_rows}x{b_columns}' ([rows]x[columns])."


class InconsistentMatrixColumnLengthError(LibraryError):
    def __init__(self: typing.Self, expected: int, received: int) -> None:
        self.message = f"Inconsistent matrix column length: expected all columns to have a length of '{expected}', received a column with a length of '{received}'."


class IncorrectActivationInputColumnCountError(ApplicationError):
    def __init__(self: typing.Self, received: int) -> None:
        self.message = f"Incorrect number of columns in activation function input: expected one, received '{received}."


class IncorrectActivationInputRowCountError(ApplicationError):
    def __init__(self: typing.Self, expected: int, received: int) -> None:
        self.message = f"Incorrect number of rows in activation function input: expected '{expected}, received '{received}."


class IncorrectArgumentTypeError(ApplicationError):
    def __init__(self: typing.Self, argument: str, correct_type: ArgumentType) -> None:
        self.message = f"Type of argument '{argument}' is incorrect: expected {'a flag' if correct_type == 'flag' else 'an option'}, received {'an option' if correct_type == 'flag' else 'a flag'}."


class IncorrectExpectedResultColumnCountError(ApplicationError):
    def __init__(self: typing.Self, received: int) -> None:
        self.message = (
            f"Incorrect number of expected result columns: expected one, received '{received}'."
        )


class IncorrectExpectedResultRowCountError(ApplicationError):
    def __init__(self: typing.Self, expected: int, received: int) -> None:
        self.message = f"Incorrect number of expected result rows: expected '{expected}', received '{received}'."


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


class IncorrectPropagationInputColumnCountError(ApplicationError):
    def __init__(self: typing.Self, received: int) -> None:
        self.message = f"Incorrect number of columns in propagation function input: expected one, received '{received}."


class IncorrectPropagationInputRowCountError(ApplicationError):
    def __init__(self: typing.Self, expected: int, received: int) -> None:
        self.message = f"Incorrect number of rows in propagation function input: expected '{expected}, received '{received}."


class IncorrectSerializedBiasDataColumnCountError(ApplicationError):
    def __init__(self: typing.Self, received: int) -> None:
        self.message = f"Incorrect number of serialized bias data columns: expected one, received '{received}'."


class IncorrectSerializedBiasDataColumnTypeError(ApplicationError):
    def __init__(self: typing.Self) -> None:
        self.message = "Type of serialized bias data column is incorrect: expected a list."


class IncorrectSerializedBiasDataRowCountError(ApplicationError):
    def __init__(self: typing.Self, expected: int, received: int) -> None:
        self.message = f"Incorrect number of serialized bias data rows: expected '{expected}', received '{received}'."


class IncorrectSerializedBiasDataTypeError(ApplicationError):
    def __init__(self: typing.Self) -> None:
        self.message = "Type of serialized bias data is incorrect: expected a list."


class IncorrectSerializedBiasDataValueTypeError(ApplicationError):
    def __init__(self: typing.Self) -> None:
        self.message = "Type of serialized bias data value is incorrect: expected a floating-point value or an integer."


class IncorrectSerializedLayerDataTypeError(ApplicationError):
    def __init__(self: typing.Self) -> None:
        self.message = "Type of serialized layer data is incorrect: expected a dictionary."


class IncorrectSerializedNetworkDataTypeError(ApplicationError):
    def __init__(self: typing.Self) -> None:
        self.message = "Type of serialized network data is incorrect: expected a list."


class IncorrectSerializedNetworkLayerCountError(ApplicationError):
    def __init__(self: typing.Self, expected: int, received: int) -> None:
        self.message = f"Incorrect number of serialized network layers: expected '{expected}', received '{received}'."


class IncorrectSerializedWeightDataColumnCountError(ApplicationError):
    def __init__(self: typing.Self, expected: int, received: int) -> None:
        self.message = f"Incorrect number of serialized weight data columns: expected '{expected}', received '{received}'."


class IncorrectSerializedWeightDataColumnTypeError(ApplicationError):
    def __init__(self: typing.Self) -> None:
        self.message = "Type of serialized weight data column is incorrect: expected a list."


class IncorrectSerializedWeightDataRowCountError(ApplicationError):
    def __init__(self: typing.Self, expected: int, received: int) -> None:
        self.message = f"Incorrect number of serialized weight data rows: expected '{expected}', received '{received}'."


class IncorrectSerializedWeightDataTypeError(ApplicationError):
    def __init__(self: typing.Self) -> None:
        self.message = "Type of serialized weight data is incorrect: expected a list."


class IncorrectSerializedWeightDataValueTypeError(ApplicationError):
    def __init__(self: typing.Self) -> None:
        self.message = "Type of serialized weight data value is incorrect: expected a floating-point value or an integer."


class InputModelFileReadError(ApplicationError):
    def __init__(self: typing.Self, error: Exception) -> None:
        self.message = f"Failed to read input model file: {str(error) or 'No message provided.'}"


class InvalidArgumentValueError(ApplicationError):
    def __init__(self: typing.Self, argument: str, reason: str, value: str) -> None:
        self.message = f"Invalid value '{value}' provided for argument '{argument}': {reason}."


class InvalidBatchSizeError(ApplicationError):
    def __init__(self: typing.Self, received: int) -> None:
        self.message = f"Invalid batch size: expected at least one, received '{received}'."


class InvalidEpochCountError(ApplicationError):
    def __init__(self: typing.Self, received: int) -> None:
        self.message = f"Invalid epoch count: expected at least one, received '{received}'."


class InvalidExpectedResultElementValueError(ApplicationError):
    def __init__(self: typing.Self, received: float) -> None:
        self.message = f"Invalid expected result element value: expected either zero or one, received '{received}'."


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


class InvalidLearningRateError(ApplicationError):
    def __init__(self: typing.Self, received: float) -> None:
        self.message = (
            f"Invalid learning rate: expected a value greater than zero, received '{received}'."
        )


class InvalidMatrixColumnCountError(LibraryError):
    def __init__(self: typing.Self, received: int) -> None:
        self.message = (
            f"Invalid number of matrix columns: expected at least one, received '{received}'."
        )


class InvalidMatrixColumnIndexError(LibraryError):
    def __init__(self: typing.Self, maximum: int, received: int) -> None:
        self.message = f"Invalid matrix column index: expected a value between zero and '{maximum}', received '{received}'."


class InvalidMatrixRowCountError(LibraryError):
    def __init__(self: typing.Self, received: int) -> None:
        self.message = (
            f"Invalid number of matrix rows: expected at least one, received '{received}'."
        )


class InvalidMatrixRowIndexError(LibraryError):
    def __init__(self: typing.Self, maximum: int, received: int) -> None:
        self.message = f"Invalid matrix row index: expected a value between zero and '{maximum}', received '{received}'."


class InvalidNetworkLayerCountError(ApplicationError):
    def __init__(self: typing.Self, received: int) -> None:
        self.message = (
            f"Invalid number of network layers: expected at least one, received '{received}'."
        )


class InvalidProgressValueError(LibraryError):
    def __init__(self: typing.Self, received: float) -> None:
        self.message = f"Invalid progress value provided: expected a value between zero and one, received '{received}'."


class MagicNumberValidationError(ApplicationError):
    def __init__(self: typing.Self, expected: int, found: int) -> None:
        self.message = f"Magic number does not match expected value: expected '{str(expected)}', found '{str(found)}'."


class MissingRequiredArgumentError(ApplicationError):
    def __init__(self: typing.Self, argument: str) -> None:
        self.message = f"Missing required argument '{argument}'."


class MissingSerializedLayerDataPropertyError(ApplicationError):
    def __init__(self: typing.Self, property: str) -> None:
        self.message = f"Missing serialized layer data property '{property}'."


class ModelDataParseError(ApplicationError):
    def __init__(self: typing.Self, error: Exception) -> None:
        self.message = f"Failed to parse model data: {str(error) or 'No message provided.'}"


class MultipleExpectedResultSelectedElementsError(ApplicationError):
    def __init__(self: typing.Self) -> None:
        self.message = "Multiple selected elements found in expected result: only one element (the desired output neuron) should be one, all others should be zero."


class OutputModelFileWriteError(ApplicationError):
    def __init__(self: typing.Self, error: Exception) -> None:
        self.message = f"Failed to write output model file: {str(error) or 'No message provided.'}"

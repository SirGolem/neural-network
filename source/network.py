import enum
import multiprocessing
import os
import parser
import pathlib
import random
import typing

import library.activation
import library.error
import library.interface
import library.matrix
import library.module
import library.type

# Types


class Argument(enum.Enum):
    BATCH_SIZE = "batch-size"
    DATASET = "dataset"
    DIRECTORY = "directory"
    EPOCH_COUNT = "epoch-count"
    LAYER_SIZES = "layer-sizes"
    LEARNING_RATE = "learning-rate"
    SAMPLE_COUNT = "sample-count"
    TEST_SAMPLE_COUNT = "test-sample-count"


Defaults = typing.TypedDict(
    "Defaults",
    {
        "batch_size": int,
        "dataset": parser.Dataset,
        "directory": pathlib.Path,
        "epoch_count": int,
        "layer_sizes": tuple[int, ...],
        "learning_rate": float,
        "sample_count": None | int,
        "test_sample_count": None | int,
    },
)


# Constants


DEFAULTS: Defaults = {
    "batch_size": 10,
    "dataset": parser.Dataset.DIGITS,
    "directory": pathlib.Path(os.getcwd()),
    "epoch_count": 10,
    "layer_sizes": (784, 25, 10),
    "learning_rate": 3,
    "sample_count": None,
    "test_sample_count": None,
}


# Classes


class Layer:
    def __init__(self: typing.Self, inputs: int, outputs: int) -> None:
        if inputs < 1:
            raise library.error.InvalidLayerInputCountError(inputs)
        if outputs < 1:
            raise library.error.InvalidLayerOutputCountError(outputs)

        self.biases = library.matrix.Matrix.from_lists(
            [[random.normalvariate() for _ in range(outputs)]]
        )
        self.inputs = inputs
        self.outputs = outputs
        self.weights = library.matrix.Matrix.from_lists(
            [[random.normalvariate() for _ in range(outputs)] for _ in range(inputs)]
        )

    def activate(
        self: typing.Self, function: typing.Callable[[float], float], inputs: library.matrix.Matrix
    ) -> library.matrix.Matrix:
        _check_activation_inputs(inputs, self.outputs)
        return inputs.apply_function_element_wise(function)

    def calculate_outputs(
        self: typing.Self, inputs: library.matrix.Matrix
    ) -> library.matrix.Matrix:
        outputs = self.propagate(inputs)
        outputs = self.activate(library.activation.sigmoid, outputs)
        return outputs

    def propagate(self: typing.Self, inputs: library.matrix.Matrix) -> library.matrix.Matrix:
        _check_propagation_inputs(inputs, self.inputs)
        return self.weights * inputs + self.biases

    def __str__(self: typing.Self) -> str:
        return f"Layer({self.inputs}, {self.outputs})"


class InputLayer(Layer):
    def __init__(self: typing.Self, neurons: int) -> None:
        super().__init__(neurons, neurons)

    def activate(
        self: typing.Self, function: typing.Callable[[float], float], inputs: library.matrix.Matrix
    ) -> library.matrix.Matrix:
        _check_activation_inputs(inputs, self.outputs)
        return inputs

    def propagate(self: typing.Self, inputs: library.matrix.Matrix) -> library.matrix.Matrix:
        _check_propagation_inputs(inputs, self.inputs)
        return inputs

    def __str__(self: typing.Self) -> str:
        return f"InputLayer({self.inputs})"


class Network:
    def __init__(self: typing.Self, layer_sizes: list[int] | tuple[int]) -> None:
        if len(layer_sizes) < 1:
            raise library.error.InvalidNetworkLayerCountError(len(layer_sizes))

        layers: list[Layer] = [InputLayer(layer_sizes[0])]

        for layer_index in range(1, len(layer_sizes)):
            layers.append(Layer(layer_sizes[layer_index - 1], layer_sizes[layer_index]))

        self.layers = tuple(layers)

    def calculate_outputs(
        self: typing.Self, inputs: library.matrix.Matrix
    ) -> library.matrix.Matrix:
        outputs = inputs

        for layer in self.layers:
            outputs = layer.calculate_outputs(outputs)

        return outputs

    def evaluate(
        self: typing.Self,
        expected_results: tuple[library.matrix.Matrix, ...],
        samples: tuple[library.matrix.Matrix, ...],
    ) -> tuple[float, int]:
        if len(expected_results) != len(samples):
            raise library.error.ExpectedResultCountDoesNotMatchSampleCountError(
                len(expected_results), len(samples)
            )

        correct = 0

        pool = multiprocessing.Pool()
        correct = sum(pool.starmap(self.__evaluate_sample__, zip(expected_results, samples)))

        return (correct / len(samples), correct)

    def train(
        self: typing.Self,
        batch_size: int,
        epoch_count: int,
        expected_results: tuple[library.matrix.Matrix, ...],
        learning_rate: float,
        samples: tuple[library.matrix.Matrix, ...],
        test_expected_results: tuple[library.matrix.Matrix, ...],
        test_samples: tuple[library.matrix.Matrix, ...],
    ) -> None:
        if batch_size < 1:
            raise library.error.InvalidBatchSizeError(batch_size)
        if epoch_count < 1:
            raise library.error.InvalidEpochCountError(epoch_count)
        if len(expected_results) != len(samples):
            raise library.error.ExpectedResultCountDoesNotMatchSampleCountError(
                len(expected_results), len(samples)
            )
        for expected_result in expected_results:
            _check_expected_result_validity(expected_result, self.layers[-1].outputs)
        if learning_rate < 0:
            raise library.error.InvalidLearningRateError(learning_rate)
        if len(test_expected_results) != len(test_samples):
            raise library.error.ExpectedResultCountDoesNotMatchSampleCountError(
                len(test_expected_results), len(test_samples)
            )
        for test_expected_result in test_expected_results:
            _check_expected_result_validity(test_expected_result, self.layers[-1].outputs)

        batch_count = len(samples) // batch_size

        for epoch_index in range(epoch_count):
            progress = library.interface.Progress(
                f"Epoch {epoch_index + 1}/{epoch_count}: ", " (Starting...)"
            )
            progress.print(0)

            epoch_data = list(zip(expected_results, samples))
            random.shuffle(epoch_data)
            epoch_expected_results = tuple(datum[0] for datum in epoch_data)
            epoch_samples = tuple(datum[1] for datum in epoch_data)

            for batch_index in range(batch_count):
                progress.suffix = f" (Batch {batch_index + 1}/{batch_count})"
                progress.print(batch_index / batch_count)
                batch_expected_results = epoch_expected_results[
                    batch_index * batch_size : (batch_index + 1) * batch_size
                ]
                batch_samples = epoch_samples[
                    batch_index * batch_size : (batch_index + 1) * batch_size
                ]
                self.__batch__(batch_expected_results, learning_rate, batch_samples)

            progress.suffix = " (Evaluating...)"
            progress.print(1)

            test_accuracy, test_correct = self.evaluate(test_expected_results, test_samples)
            train_accuracy, train_correct = self.evaluate(expected_results, samples)
            progress.complete(
                f"{progress.prefix}{test_accuracy * 100}% ({test_correct}/{len(test_samples)}) [Testing Split], {train_accuracy * 100}% ({train_correct}/{len(samples)}) [Training Split]."
            )

    def __backpropagate__(
        self: typing.Self, expected_outputs: library.matrix.Matrix, inputs: library.matrix.Matrix
    ) -> tuple[tuple[library.matrix.Matrix, ...], tuple[library.matrix.Matrix, ...]]:
        activations = [inputs]
        weighted_inputs: list[library.matrix.Matrix] = []

        for layer in self.layers[1:]:
            weighted_inputs.append(layer.propagate(activations[-1]))
            activations.append(layer.activate(library.activation.sigmoid, weighted_inputs[-1]))

        errors = [
            (activations[-1] - expected_outputs).multiply_element_wise(
                weighted_inputs[-1].apply_function_element_wise(
                    library.activation.sigmoid_derivative
                )
            )
        ]

        for layer_index in range(len(self.layers) - 2, 0, -1):
            errors.insert(
                0,
                (
                    self.layers[layer_index + 1].weights.transpose() * errors[0]
                ).multiply_element_wise(
                    weighted_inputs[layer_index - 1].apply_function_element_wise(
                        library.activation.sigmoid_derivative
                    )
                ),
            )

        weight_partial_derivatives: list[library.matrix.Matrix] = []

        for layer_index in range(1, len(self.layers)):
            layer_weight_partial_derivatives = library.matrix.Matrix(
                self.layers[layer_index].weights.columns, self.layers[layer_index].weights.rows
            )

            for column_index in range(self.layers[layer_index].weights.columns):
                for row_index in range(self.layers[layer_index].weights.rows):
                    layer_weight_partial_derivatives.set(
                        column_index,
                        row_index,
                        (
                            activations[layer_index - 1].get(0, column_index)
                            * errors[layer_index - 1].get(0, row_index)
                        ),
                    )

            weight_partial_derivatives.append(layer_weight_partial_derivatives)

        return (tuple(errors), tuple(weight_partial_derivatives))

    def __batch__(
        self: typing.Self,
        expected_results: tuple[library.matrix.Matrix, ...],
        learning_rate: float,
        samples: tuple[library.matrix.Matrix, ...],
    ) -> None:
        biases_gradient = [library.matrix.Matrix(1, layer.outputs) for layer in self.layers[1:]]
        weights_gradient = [
            library.matrix.Matrix(layer.inputs, layer.outputs) for layer in self.layers[1:]
        ]

        for expected_result, sample in zip(expected_results, samples):
            biases_gradient_change, weights_gradient_change = self.__backpropagate__(
                expected_result, sample
            )

            for layer_index in range(len(self.layers) - 1):
                biases_gradient[layer_index] += biases_gradient_change[layer_index]
                weights_gradient[layer_index] += weights_gradient_change[layer_index]

        for layer_index in range(1, len(self.layers)):
            self.layers[layer_index].biases -= biases_gradient[layer_index - 1] * (
                learning_rate / len(samples)
            )
            self.layers[layer_index].weights -= weights_gradient[layer_index - 1] * (
                learning_rate / len(samples)
            )

    def __evaluate_sample__(
        self: typing.Self, expected_result: library.matrix.Matrix, sample: library.matrix.Matrix
    ) -> bool:
        _check_expected_result_validity(expected_result, self.layers[-1].outputs)

        _, expected_row, _ = expected_result.get_maximum_value()
        result = self.calculate_outputs(sample)
        _, result_row, _ = result.get_maximum_value()

        return expected_row == result_row

    def __str__(self: typing.Self) -> str:
        return f"Network({len(self.layers)}) {{{''.join(['\n  ' + str(layer) for layer in self.layers])}\n}}"


# Functions


def _check_activation_inputs(inputs: library.matrix.Matrix, layer_outputs: int) -> None:
    if inputs.columns != 1:
        raise library.error.IncorrectActivationInputColumnCountError(inputs.columns)
    if inputs.rows != layer_outputs:
        raise library.error.IncorrectActivationInputRowCountError(layer_outputs, inputs.rows)


def _check_expected_result_validity(
    expected_result: library.matrix.Matrix, network_output_count: int
) -> None:
    if expected_result.columns != 1:
        raise library.error.IncorrectExpectedResultColumnCountError(expected_result.columns)
    if expected_result.rows != network_output_count:
        raise library.error.IncorrectExpectedResultRowCountError(
            network_output_count, expected_result.rows
        )

    found_one = False

    for row_index in range(expected_result.rows):
        value = expected_result.get(0, row_index)

        if value != 0 and value != 1:
            raise library.error.InvalidExpectedResultElementValueError(value)
        if found_one and value == 1:
            raise library.error.MultipleExpectedResultSelectedElementsError
        if value == 1:
            found_one = True


def _check_propagation_inputs(inputs: library.matrix.Matrix, layer_inputs: int) -> None:
    if inputs.columns != 1:
        raise library.error.IncorrectPropagationInputColumnCountError(inputs.columns)
    if inputs.rows != layer_inputs:
        raise library.error.IncorrectPropagationInputRowCountError(layer_inputs, inputs.rows)


# Main Program


def main() -> bool:
    arguments = library.interface.parse_arguments()

    batch_size_option = arguments.get_option(Argument.BATCH_SIZE.value, str(DEFAULTS["batch_size"]))

    try:
        batch_size = int(batch_size_option)
        if batch_size < 1:
            raise ValueError
    except ValueError:
        raise library.error.InvalidArgumentValueError(
            Argument.BATCH_SIZE.value, "expected an integer greater than zero", batch_size_option
        )

    dataset_option = arguments.get_option(Argument.DATASET.value, DEFAULTS["dataset"].value)

    try:
        dataset = parser.Dataset(dataset_option)
    except ValueError:
        raise library.error.InvalidArgumentValueError(
            Argument.DATASET.value,
            f"expected one of {', '.join([f"'{value.value}'" for value in parser.Dataset._member_map_.values()])}",
            dataset_option,
        )

    directory_option = arguments.get_option(Argument.DIRECTORY.value, str(DEFAULTS["directory"]))
    directory = (pathlib.Path(directory_option)).expanduser()

    if not directory.exists():
        raise library.error.InvalidArgumentValueError(
            Argument.DIRECTORY.value, "path does not exist", directory_option
        )

    if not directory.is_dir():
        raise library.error.InvalidArgumentValueError(
            Argument.DIRECTORY.value, "path is not a directory", directory_option
        )

    epoch_count_option = arguments.get_option(
        Argument.EPOCH_COUNT.value, str(DEFAULTS["epoch_count"])
    )

    try:
        epoch_count = int(epoch_count_option)
        if epoch_count < 1:
            raise ValueError
    except ValueError:
        raise library.error.InvalidArgumentValueError(
            Argument.EPOCH_COUNT.value, "expected an integer greater than zero", epoch_count_option
        )

    layer_sizes_option = arguments.get_option(
        Argument.LAYER_SIZES.value,
        ",".join([str(layer_size) for layer_size in DEFAULTS["layer_sizes"]]),
    )

    try:
        layer_sizes = [
            int(layer_size.strip())
            for layer_size in layer_sizes_option.split(",")
            if len(layer_size) > 0
        ]

        if len(layer_sizes) == 0:
            raise ValueError
    except ValueError:
        raise library.error.InvalidArgumentValueError(
            Argument.LAYER_SIZES.value,
            "expected a comma-separated list of integers",
            layer_sizes_option,
        )

    learning_rate_option = arguments.get_option(
        Argument.LEARNING_RATE.value, str(DEFAULTS["learning_rate"])
    )

    try:
        learning_rate = float(learning_rate_option)
        if learning_rate <= 0:
            raise ValueError
    except ValueError:
        raise library.error.InvalidArgumentValueError(
            Argument.LEARNING_RATE.value,
            "expected a floating-point value greater than zero",
            learning_rate_option,
        )

    sample_count_option = arguments.get_option(
        Argument.SAMPLE_COUNT.value, str(DEFAULTS["sample_count"])
    )

    try:
        if sample_count_option == str(None):
            sample_count = None
        else:
            sample_count = int(sample_count_option)
            if sample_count < 1:
                raise ValueError
    except ValueError:
        raise library.error.InvalidArgumentValueError(
            Argument.SAMPLE_COUNT.value,
            "expected an integer greater than zero",
            sample_count_option,
        )

    test_sample_count_option = arguments.get_option(
        Argument.TEST_SAMPLE_COUNT.value, str(DEFAULTS["test_sample_count"])
    )

    try:
        if test_sample_count_option == str(None):
            test_sample_count = None
        else:
            test_sample_count = int(test_sample_count_option)
            if test_sample_count < 1:
                raise ValueError
    except ValueError:
        raise library.error.InvalidArgumentValueError(
            Argument.TEST_SAMPLE_COUNT.value,
            "expected an integer greater than zero",
            test_sample_count_option,
        )

    print("Creating network...", end="", flush=True)
    network = Network(layer_sizes)
    library.interface.clear_line()
    print("Created network.")

    (_, _, _, images) = parser.parse_images(
        dataset, parser.DatasetSplit.TRAIN, directory, sample_count
    )
    (_, _, _, test_images) = parser.parse_images(
        dataset, parser.DatasetSplit.TEST, directory, test_sample_count
    )
    label_mappings = parser.parse_label_mappings(dataset, directory)
    (_, labels) = parser.parse_labels(
        dataset, parser.DatasetSplit.TRAIN, directory, label_mappings, sample_count
    )
    (_, test_labels) = parser.parse_labels(
        dataset, parser.DatasetSplit.TEST, directory, label_mappings, test_sample_count
    )

    expected_results = parser.convert_labels_to_matrices(labels)
    samples = parser.convert_images_to_matrices(images)
    test_expected_results = parser.convert_labels_to_matrices(test_labels)
    test_samples = parser.convert_images_to_matrices(test_images)

    print("Evaluating network...", end="", flush=True)
    test_accuracy, test_correct = network.evaluate(test_expected_results, test_samples)
    train_accuracy, train_correct = network.evaluate(expected_results, samples)
    library.interface.clear_line()
    print(
        f"Evaluated network: {test_accuracy * 100}% ({test_correct}/{len(test_images)}) [Testing Split], {train_accuracy * 100}% ({train_correct}/{len(images)}) [Training Split]."
    )

    print(
        f"Training network (batch size: {batch_size}, epoch count: {epoch_count}, learning rate: {learning_rate})..."
    )
    network.train(
        batch_size,
        epoch_count,
        expected_results,
        learning_rate,
        samples,
        test_expected_results,
        test_samples,
    )
    print("Trained network.")

    return True


if __name__ == "__main__":
    library.module.try_main(main)

import enum
import itertools
import parser
import pathlib
import random
import typing

import library.activation
import library.cost
import library.error
import library.interface
import library.module
import library.type

# Types


class Argument(enum.Enum):
    LAYERS = "layers"


# Constants


NULL_EXPECTED_OUTPUTS: list[int] = [0] * 9


# Classes


class Layer:
    def __init__(self: typing.Self, inputs: int, outputs: int) -> None:
        if inputs < 1:
            raise library.error.InvalidLayerInputCountError(inputs)
        if outputs < 1:
            raise library.error.InvalidLayerOutputCountError(outputs)

        self.biases = tuple([random.uniform(-1, 1) for _ in range(outputs)])
        self.inputs = inputs
        self.outputs = outputs
        self.weights = tuple(
            [tuple([random.uniform(-1, 1) for _ in range(outputs)]) for _ in range(inputs)]
        )

    def calculate_outputs(self: typing.Self, inputs: tuple[float, ...]) -> tuple[float, ...]:
        if len(inputs) != self.inputs:
            raise library.error.IncorrectLayerInputCountError(self.inputs, len(inputs))

        outputs = self.__propagate__(inputs)
        outputs = self.__activate__(library.activation.sigmoid, outputs)
        return outputs

    def __activate__(
        self: typing.Self, function: typing.Callable[[float], float], inputs: tuple[float, ...]
    ) -> tuple[float, ...]:
        outputs: list[float] = []

        for input in inputs:
            outputs.append(function(input))

        return tuple(outputs)

    def __propagate__(self: typing.Self, inputs: tuple[float, ...]) -> tuple[float, ...]:
        outputs: list[float] = []

        for output in range(self.outputs):
            weighted_input = 0.0

            for input in range(self.inputs):
                weighted_input += inputs[input] * self.weights[input][output]

            weighted_input += self.biases[output]
            outputs.append(weighted_input)

        return tuple(outputs)

    def __str__(self: typing.Self) -> str:
        return f"Layer({self.inputs}, {self.outputs})"


class InputLayer(Layer):
    def __init__(self: typing.Self, size: int) -> None:
        super().__init__(size, size)

    def calculate_outputs(self: typing.Self, inputs: tuple[float, ...]) -> tuple[float, ...]:
        if len(inputs) != self.inputs:
            raise library.error.IncorrectLayerInputCountError(self.inputs, len(inputs))

        return inputs

    def __str__(self: typing.Self) -> str:
        return f"InputLayer({self.inputs})"


class Network:
    def __init__(self: typing.Self, layer_sizes: list[int]) -> None:
        if len(layer_sizes) < 1:
            raise library.error.InvalidNetworkLayerCountError(len(layer_sizes))

        layers: list[Layer] = [InputLayer(layer_sizes[0])]

        for layer in range(1, len(layer_sizes)):
            layers.append(Layer(layer_sizes[layer - 1], layer_sizes[layer]))

        self.layers = tuple(layers)

    def calculate_cost(
        self: typing.Self,
        inputs: tuple[library.type.TupleImage, ...],
        labels: tuple[library.type.Label, ...],
    ) -> float:
        if len(inputs) != len(labels):
            raise library.error.ImageCountDoesNotMatchLabelCountError(len(inputs), len(labels))

        expected_outputs = tuple(
            tuple(NULL_EXPECTED_OUTPUTS[: int(label)] + [1] + NULL_EXPECTED_OUTPUTS[int(label) :])
            for label in labels
        )
        outputs = tuple(
            self.calculate_outputs(tuple(itertools.chain.from_iterable(input))) for input in inputs
        )

        try:
            return library.cost.mean_squared(expected_outputs, len(inputs), tuple(outputs))
        except ValueError as error:
            raise library.error.CostCalculationError(error)

    def calculate_outputs(self: typing.Self, inputs: tuple[float, ...]) -> tuple[float, ...]:
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)

        return inputs

    def __str__(self: typing.Self) -> str:
        return f"Network({len(self.layers)}) {{{''.join(['\n  ' + str(layer) for layer in self.layers])}\n}}"


# Main Program


def main() -> bool:
    arguments = library.interface.parse_arguments()

    layers_option = arguments.get_option(Argument.LAYERS.value)

    try:
        layer_sizes = [
            int(layer_size.strip())
            for layer_size in layers_option.split(",")
            if len(layer_size) > 0
        ]

        if len(layer_sizes) == 0:
            raise ValueError
    except ValueError:
        raise library.error.InvalidArgumentValueError(
            Argument.LAYERS.value, "expected a comma-separated list of integers", layers_option
        )

    print("Creating network...")
    network = Network(layer_sizes)
    print("Created network.")

    (_, _, _, images) = parser.parse_images(
        parser.DEFAULTS["dataset"],
        parser.DEFAULTS["dataset_split"],
        pathlib.Path("~/Downloads/emnist").expanduser(),  # parser.DEFAULTS["directory"]
        100,  # parser.DEFAULTS["count"]
    )
    label_mappings = parser.parse_label_mappings(
        parser.DEFAULTS["dataset"],
        pathlib.Path("~/Downloads/emnist").expanduser(),  # parser.DEFAULTS["directory"]
    )
    (_, labels) = parser.parse_labels(
        parser.DEFAULTS["dataset"],
        parser.DEFAULTS["dataset_split"],
        pathlib.Path("~/Downloads/emnist").expanduser(),  # parser.DEFAULTS["directory"]
        label_mappings,
        100,  # parser.DEFAULTS["count"]
    )

    print("Calculating network cost...")
    cost = network.calculate_cost(images, labels)
    print("Calculated network cost.")
    print(cost)

    return True


if __name__ == "__main__":
    library.module.try_main(main)

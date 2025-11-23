import decimal
import math

import library
import library.error
import library.type

# Activation Functions (https://en.wikipedia.org/wiki/Activation_function#Table_of_activation_functions)


def hyperbolic_tangent(input: float) -> float:
    return ((math.e**input) - (math.e ** (-input))) / ((math.e**input) + (math.e ** (-input)))


def hyperbolic_tangent_derivative(input: float) -> float:
    return 1 - (hyperbolic_tangent(input) ** 2)


def identity(input: float) -> float:
    return input


def identity_derivative(_: float) -> float:
    return 1


def rectified_linear_unit(input: float) -> float:
    return max(0, input)


def rectified_linear_unit_derivative(input: float) -> float:
    return 0 if input <= 0 else 1


def sigmoid(input: float) -> float:
    return 1 / (1 + (math.e ** (-input)))


def sigmoid_derivative(input: float) -> float:
    return sigmoid(input) * (1 - sigmoid(input))


def softplus(input: float) -> float:
    return math.log(1 + (decimal.Decimal(math.e) ** decimal.Decimal(input)), math.e)


def softplus_derivative(input: float) -> float:
    return float(1 / (1 + (decimal.Decimal(math.e) ** (-decimal.Decimal(input)))))


def step(input: float) -> float:
    return 0 if input < 0 else 1


def step_derivative(_: float) -> float:
    return 0


# Constants


ACTIVATION_FUNCTIONS: tuple[library.type.ActivationFunctionPair, ...] = (
    {
        "derivative": hyperbolic_tangent_derivative,
        "function": hyperbolic_tangent,
        "identifier": library.type.ActivationFunctionIdentifier.HYPERBOLIC_TANGENT,
    },
    {
        "derivative": identity_derivative,
        "function": identity,
        "identifier": library.type.ActivationFunctionIdentifier.IDENTITY,
    },
    {
        "derivative": rectified_linear_unit_derivative,
        "function": rectified_linear_unit,
        "identifier": library.type.ActivationFunctionIdentifier.RECTIFIED_LINEAR_UNIT,
    },
    {
        "derivative": sigmoid_derivative,
        "function": sigmoid,
        "identifier": library.type.ActivationFunctionIdentifier.SIGMOID,
    },
    {
        "derivative": softplus_derivative,
        "function": softplus,
        "identifier": library.type.ActivationFunctionIdentifier.SOFTPLUS,
    },
    {
        "derivative": step_derivative,
        "function": step,
        "identifier": library.type.ActivationFunctionIdentifier.STEP,
    },
)


# Functions


def get_activation_function_pair(
    identifier: library.type.ActivationFunctionIdentifier,
) -> library.type.ActivationFunctionPair:
    activation_function_pairs = tuple(
        filter(
            lambda activation_function_pair: activation_function_pair["identifier"] == identifier,
            list(ACTIVATION_FUNCTIONS),
        )
    )
    if len(activation_function_pairs) != 1:
        raise library.error.DuplicateActivationFunctionPairIdentifierError(identifier.value)

    return activation_function_pairs[0]

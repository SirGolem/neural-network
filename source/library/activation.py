import math

# Functions


def sigmoid(input: float) -> float:
    return 1 / (1 + math.e ** (-input))


def sigmoid_derivative(input: float) -> float:
    return sigmoid(input) * (1 - sigmoid(input))


def step(input: float) -> float:
    return 0 if input <= 0 else 1

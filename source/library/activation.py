import math

# Functions


def sigmoid(input: float) -> float:
    return 1 / (1 + math.e ** (-input))


def step(input: float) -> float:
    return 0 if input <= 0 else 1

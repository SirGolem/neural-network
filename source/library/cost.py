# Functions


def mean_squared(
    expected_outputs: tuple[tuple[int, ...], ...],
    inputs: int,
    outputs: tuple[tuple[float, ...], ...],
) -> float:
    if inputs < 1:
        raise ValueError(f"Invalid number of inputs: expected at least one, received '{inputs}'.")
    if inputs != len(expected_outputs) or inputs != len(outputs):
        raise ValueError(
            f"Expected output count ({len(expected_outputs)}) does not match output count ({len(outputs)})."
        )

    sum_of_squares = 0.0

    for input in range(inputs):
        if len(expected_outputs[input]) != len(outputs[input]):
            raise ValueError(
                f"Expected output length ({len(expected_outputs[input])}) does not match output length ({len(outputs[input])})."
            )

        sum_of_squares += sum(
            (expected_outputs[input][value] - outputs[input][value]) ** 2
            for value in range(len(expected_outputs[input]))
        )

    return 1 / (2 * inputs) * sum_of_squares

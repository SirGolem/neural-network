from __future__ import annotations

import typing

import library.error
import library.type

# Classes


class Matrix:
    def __init__(self: typing.Self, columns: int, rows: int) -> None:
        if columns < 1:
            raise library.error.InvalidMatrixColumnCountError(columns)
        if rows < 1:
            raise library.error.InvalidMatrixRowCountError(rows)

        self.columns = columns
        self.data = [[0.0 for _ in range(rows)] for _ in range(columns)]
        self.rows = rows

    def add(self: typing.Self, other: Matrix) -> Matrix:
        if other.columns != self.columns or other.rows != self.rows:
            raise library.error.IncompatibleMatrixDimensionsError(
                self.columns, self.rows, other.columns, other.rows
            )

        matrix = Matrix(self.columns, self.rows)

        for column_index in range(matrix.columns):
            for row_index in range(matrix.rows):
                matrix.set(
                    column_index,
                    row_index,
                    other.get(column_index, row_index) + self.get(column_index, row_index),
                )

        return matrix

    def apply_function_element_wise(
        self: typing.Self, function: typing.Callable[[float], float]
    ) -> Matrix:
        matrix = Matrix(self.columns, self.rows)

        for column_index in range(matrix.columns):
            for row_index in range(matrix.rows):
                matrix.set(column_index, row_index, function(self.get(column_index, row_index)))

        return matrix

    @classmethod
    def from_lists(matrix_class: typing.Type[typing.Self], lists: list[list[float]]) -> typing.Self:
        columns = len(lists)
        if columns < 1:
            raise library.error.InvalidMatrixColumnCountError(columns)
        rows = len(lists[0])
        if rows < 1:
            raise library.error.InvalidMatrixRowCountError(rows)

        for column in lists:
            if len(column) != rows:
                raise library.error.InconsistentMatrixColumnLengthError(rows, len(column))

        matrix = matrix_class(columns, rows)
        matrix.data = [column.copy() for column in lists]
        return matrix

    @classmethod
    def from_tuples(
        matrix_class: typing.Type[typing.Self], tuples: tuple[tuple[float, ...], ...]
    ) -> typing.Self:
        columns = len(tuples)
        if columns < 1:
            raise library.error.InvalidMatrixColumnCountError(columns)
        rows = len(tuples[0])
        if rows < 1:
            raise library.error.InvalidMatrixRowCountError(rows)

        for column in tuples:
            if len(column) != rows:
                raise library.error.InconsistentMatrixColumnLengthError(rows, len(column))

        matrix = matrix_class(columns, rows)
        matrix.data = [list(column) for column in tuples]
        return matrix

    def get(self: typing.Self, column: int, row: int) -> float:
        if column < 0 or column >= self.columns:
            raise library.error.InvalidMatrixColumnIndexError(self.columns - 1, column)
        if row < 0 or row >= self.rows:
            raise library.error.InvalidMatrixRowIndexError(self.rows - 1, row)

        return self.data[column][row]

    def get_maximum_value(self: typing.Self) -> tuple[int, int, float]:
        maximum_column = -1
        maximum_row = -1
        maximum_value = 0.0

        for column_index in range(self.columns):
            for row_index in range(self.rows):
                value = self.get(column_index, row_index)

                if value > maximum_value:
                    maximum_column = column_index
                    maximum_row = row_index
                    maximum_value = value

        return (maximum_column, maximum_row, maximum_value)

    @classmethod
    def identity(matrix_class: typing.Type[typing.Self], size: int) -> Matrix:
        if size < 1:
            raise library.error.InvalidMatrixColumnCountError(size)

        matrix = matrix_class(size, size)

        for index in range(size):
            matrix.set(index, index, 1)

        return matrix

    def multiply_by_matrix(self: typing.Self, other: Matrix) -> Matrix:
        if other.rows != self.columns:
            raise library.error.IncompatibleMatrixDimensionsError(
                self.columns, self.rows, other.columns, other.rows
            )

        matrix = Matrix(other.columns, self.rows)

        for column_index in range(matrix.columns):
            for row_index in range(matrix.rows):
                value = 0.0

                for index in range(self.columns):
                    value += self.get(index, row_index) * other.get(column_index, index)

                matrix.set(column_index, row_index, value)

        return matrix

    def multiply_by_scalar(self: typing.Self, other: float) -> Matrix:
        matrix = Matrix(self.columns, self.rows)

        for column_index in range(matrix.columns):
            for row_index in range(matrix.rows):
                matrix.set(column_index, row_index, other * self.get(column_index, row_index))

        return matrix

    def multiply_element_wise(self: typing.Self, other: Matrix) -> Matrix:
        if other.columns != self.columns or other.rows != self.rows:
            raise library.error.IncompatibleMatrixDimensionsError(
                self.columns, self.rows, other.columns, other.rows
            )

        matrix = Matrix(self.columns, self.rows)

        for column_index in range(matrix.columns):
            for row_index in range(matrix.rows):
                matrix.set(
                    column_index,
                    row_index,
                    other.get(column_index, row_index) * self.get(column_index, row_index),
                )

        return matrix

    def set(self: typing.Self, column: int, row: int, value: float) -> None:
        if column < 0 or column >= self.columns:
            raise library.error.InvalidMatrixColumnIndexError(self.columns - 1, column)
        if row < 0 or row >= self.rows:
            raise library.error.InvalidMatrixRowIndexError(self.rows - 1, row)

        self.data[column][row] = value

    def subtract(self: typing.Self, other: Matrix) -> Matrix:
        if other.columns != self.columns or other.rows != self.rows:
            raise library.error.IncompatibleMatrixDimensionsError(
                self.columns, self.rows, other.columns, other.rows
            )

        matrix = Matrix(self.columns, self.rows)

        for column_index in range(matrix.columns):
            for row_index in range(matrix.rows):
                matrix.set(
                    column_index,
                    row_index,
                    self.get(column_index, row_index) - other.get(column_index, row_index),
                )

        return matrix

    def sum_of_elements(self: typing.Self) -> float:
        sum = 0.0

        for column_index in range(self.columns):
            for row_index in range(self.rows):
                sum += self.get(column_index, row_index)

        return sum

    def to_string(self: typing.Self) -> str:
        column_lengths: list[int] = []

        for column_index in range(self.columns):
            column_length = 0

            for row_index in range(self.rows):
                element_length = len(str(self.get(column_index, row_index)))

                if element_length > column_length:
                    column_length = element_length

            column_lengths.append(column_length)

        row_strings: list[str] = []

        for row_index in range(self.rows):
            row_string = (
                "["
                if row_index == 0 and self.rows == 1
                else "⎡"
                if row_index == 0
                else "⎣"
                if row_index == self.rows - 1
                else "⎢"
            )

            elements: list[str] = []

            for column_index in range(self.columns):
                elements.append(
                    str(self.get(column_index, row_index)).ljust(column_lengths[column_index])
                )

            row_string += " ".join(elements)
            row_string += (
                "]"
                if row_index == 0 and self.rows == 1
                else "⎤"
                if row_index == 0
                else "⎦"
                if row_index == self.rows - 1
                else "⎥"
            )
            row_strings.append(row_string)

        return "\n".join(row_strings)

    def transpose(self: typing.Self) -> Matrix:
        transposed = Matrix(self.rows, self.columns)

        for column_index in range(self.columns):
            for row_index in range(self.rows):
                transposed.set(row_index, column_index, self.get(column_index, row_index))

        return transposed

    def __add__(self: typing.Self, other: Matrix) -> Matrix:
        return self.add(other)

    def __mul__(self: typing.Self, other: Matrix | float) -> Matrix:
        if isinstance(other, Matrix):
            return self.multiply_by_matrix(other)
        else:
            return self.multiply_by_scalar(other)

    def __str__(self: typing.Self) -> str:
        return self.to_string()

    def __sub__(self: typing.Self, other: Matrix) -> Matrix:
        return self.subtract(other)

from __future__ import annotations

import typing

import library.data
import library.error

# Classes


class Matrix:
    def __init__(self: typing.Self, columns: int, rows: int) -> None:
        if columns < 1:
            raise library.error.InvalidMatrixColumnCountError(columns)
        if rows < 1:
            raise library.error.InvalidMatrixRowCountError(rows)

        self.columns = columns
        self.data: list[list[library.data.Number]] = [
            [0 for _ in range(rows)] for _ in range(columns)
        ]
        self.rows = rows

    @staticmethod
    def add(a: Matrix, b: Matrix) -> Matrix:
        if a.columns != b.columns or a.rows != b.rows:
            raise library.error.IncompatibleMatrixDimensionsError(
                a.columns, a.rows, b.columns, b.rows
            )

        matrix = Matrix(a.columns, a.rows)

        for column in range(matrix.columns):
            for row in range(matrix.rows):
                matrix.set(column, row, a.get(column, row) + b.get(column, row))

        return matrix

    @classmethod
    def from_tuple(
        matrix_class: typing.Type[typing.Self], tuple: tuple[tuple[library.data.Number, ...], ...]
    ) -> typing.Self:
        columns = len(tuple)
        if columns < 1:
            raise library.error.InvalidMatrixColumnCountError(columns)
        rows = len(tuple[0])
        if rows < 1:
            raise library.error.InvalidMatrixRowCountError(rows)

        for column in tuple:
            if len(column) != rows:
                raise library.error.InconsistentMatrixColumnLengthError(rows, len(column))

        matrix = matrix_class(columns, rows)
        matrix.data = [list(column) for column in tuple]
        return matrix

    def get(self: typing.Self, column: int, row: int) -> library.data.Number:
        if column < 0 or column >= self.columns:
            raise library.error.InvalidMatrixColumnIndexError(self.columns - 1, column)
        if row < 0 or row >= self.rows:
            raise library.error.InvalidMatrixRowIndexError(self.rows - 1, row)

        return self.data[column][row]

    @staticmethod
    def multiply_by_matrix(a: Matrix, b: Matrix) -> Matrix:
        if a.columns != b.rows:
            raise library.error.IncompatibleMatrixDimensionsError(
                a.columns, a.rows, b.columns, b.rows
            )

        matrix = Matrix(b.columns, a.rows)

        for column in range(matrix.columns):
            for row in range(matrix.rows):
                value: library.data.Number = 0

                for index in range(a.columns):
                    value += a.get(index, row) * b.get(column, index)

                matrix.set(column, row, value)

        return matrix

    @staticmethod
    def multiply_by_scalar(a: Matrix, b: library.data.Number) -> Matrix:
        matrix = Matrix(a.columns, a.rows)

        for column in range(matrix.columns):
            for row in range(matrix.rows):
                matrix.set(column, row, a.get(column, row) * b)

        return matrix

    def set(self: typing.Self, column: int, row: int, value: library.data.Number) -> None:
        if column < 0 or column >= self.columns:
            raise library.error.InvalidMatrixColumnIndexError(self.columns - 1, column)
        if row < 0 or row >= self.rows:
            raise library.error.InvalidMatrixRowIndexError(self.rows - 1, row)

        self.data[column][row] = value

    @staticmethod
    def subtract(a: Matrix, b: Matrix) -> Matrix:
        if a.columns != b.columns or a.rows != b.rows:
            raise library.error.IncompatibleMatrixDimensionsError(
                a.columns, a.rows, b.columns, b.rows
            )

        matrix = Matrix(a.columns, a.rows)

        for column in range(matrix.columns):
            for row in range(matrix.rows):
                matrix.set(column, row, a.get(column, row) - b.get(column, row))

        return matrix

    @staticmethod
    def to_string(matrix: Matrix) -> str:
        column_lengths: list[int] = []

        for column in range(matrix.columns):
            column_length = 0

            for row in range(matrix.rows):
                element_length = len(str(matrix.get(column, row)))

                if element_length > column_length:
                    column_length = element_length

            column_lengths.append(column_length)

        row_strings: list[str] = []

        for row in range(matrix.rows):
            row_string = (
                "["
                if row == 0 and matrix.rows == 1
                else "⎡"
                if row == 0
                else "⎣"
                if row == matrix.rows - 1
                else "⎢"
            )

            elements: list[str] = []

            for column in range(matrix.columns):
                elements.append(str(matrix.get(column, row)).ljust(column_lengths[column]))

            row_string += " ".join(elements)
            row_string += (
                "]"
                if row == 0 and matrix.rows == 1
                else "⎤"
                if row == 0
                else "⎦"
                if row == matrix.rows - 1
                else "⎥"
            )
            row_strings.append(row_string)

        return "\n".join(row_strings)

    @staticmethod
    def transpose(matrix: Matrix) -> Matrix:
        transposed = Matrix(matrix.rows, matrix.columns)

        for column in range(matrix.columns):
            for row in range(matrix.rows):
                transposed.set(row, column, matrix.get(column, row))

        return transposed

    def transposed(self: typing.Self) -> Matrix:
        return Matrix.transpose(self)

    @classmethod
    def vector(matrix_class: typing.Type[typing.Self], rows: int) -> typing.Self:
        return matrix_class(1, rows)

    def __add__(self: typing.Self, other: Matrix) -> Matrix:
        return Matrix.add(self, other)

    def __mul__(self: typing.Self, other: Matrix | library.data.Number) -> Matrix:
        if isinstance(other, Matrix):
            return Matrix.multiply_by_matrix(self, other)
        else:
            return Matrix.multiply_by_scalar(self, other)

    def __str__(self: typing.Self) -> str:
        return Matrix.to_string(self)

    def __sub__(self: typing.Self, other: Matrix) -> Matrix:
        return Matrix.subtract(self, other)

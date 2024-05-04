from __future__ import annotations
import copy


class Matrix:
    def __init__(self, data: list[list[int]]):
        self.raw_data = data
        self.rows = len(data)
        self.cols = len(data[0])

    def __mul__(self, other: 'Matrix' | int) -> 'Matrix':
        if isinstance(other, int):
            return self._constant_multiplication(other)
        if isinstance(other, Matrix):
            return self._matrix_multiplication(other)
        raise TypeError(f'Unsupported operand of type {type(other)}')

    def _constant_multiplication(self, other: int) -> 'Matrix':
        data_copy = [[i * other for i in j] for j in self.raw_data]
        return Matrix(data_copy)

    def _matrix_multiplication(self, other: 'Matrix') -> 'Matrix':
        if self.cols != other.rows:
            raise ValueError('Incompatible matrix dimensions')

        result = [[self._dot_product(row, col) for col in other.get_cols()] for row in self.raw_data]

        return Matrix(result)

    @staticmethod
    def _dot_product(row: list[int], col: list[int]) -> int:
        if len(row) != len(col):
            raise ValueError('Incompatible matrix dimensions')

        return sum([row[i] * col[i] for i in range(len(row))])

    def get_rows(self) -> list[list[int]]:
        return copy.deepcopy(self.raw_data)

    def get_cols(self) -> list[list[int]]:
        return [[self.raw_data[j][i] for j in range(len(self.raw_data))] for i in range(len(self.raw_data[0]))]

    def __getitem__(self, index: int) -> list[int]:
        return self.raw_data[index]

    @staticmethod
    def zeros(size: int) -> 'Matrix':
        return Matrix([[0] * size for _ in range(size)])

    @staticmethod
    def add_diag(matrix: 'Matrix', offset: int, value: int) -> 'Matrix':
        dx, dy = [-abs(offset), 0] if offset > 0 else [0, -abs(offset)]

        for i in range(abs(offset), matrix.rows):
            matrix[i + dx][i + dy] = value

        return matrix

    @staticmethod
    def diag(value: int, offset: int, size: int) -> 'Matrix':
        matrix = Matrix.zeros(size)
        return Matrix.add_diag(matrix, offset, value)


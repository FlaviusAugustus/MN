import math
from Matrix import Matrix


# 193579
def equation_a() -> tuple[Matrix, list[float]]:
    size = 9*9*7

    a = Matrix.zeros(size)
    Matrix.add_diag(a, 0, 5)
    Matrix.add_diag(a, 1, -1)
    Matrix.add_diag(a, -1, -1)
    Matrix.add_diag(a, -2, -1)
    Matrix.add_diag(a, 2, -1)

    b = [math.sin(i*6) for i in range(size)]

    return a, b

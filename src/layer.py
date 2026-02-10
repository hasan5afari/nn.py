# Implementation of a single layer consisting of 3 neurons


Matrix = list[list[float]]
Vector = list[float]


def dimension(matrix: Matrix) -> tuple[int, int]:
    return len(matrix), len(matrix[0])


def transpose(matrix: Matrix) -> Matrix:
    r, c = dimension(matrix)
    transposed_matrix = [[0 for _ in range(r)] for _ in range(c)]

    for column in range(c):
        for row in range(r):
            transposed_matrix[column][row] = matrix[row][column]

    return transposed_matrix


def dot(matrix_a: Matrix, matrix_b: Matrix) -> Matrix:
    r_a, c_a = dimension(matrix_a)
    r_b, c_b = dimension(matrix_b)

    assert c_a == r_b, f"Dot product cannot be performed on matrices with size: ({r_a},{c_a}), ({r_b},{c_b})"

    dot_product = [[0 for _ in range(r_a)] for _ in range(c_b)]

    for i in range(r_a):
        for j in range(c_b):
            for k in range(r_b):
                dot_product[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return dot_product


def add_vector(matrix: Matrix, vector: Vector) -> Matrix:
    r_m, c_m = dimension(matrix)
    v_c = len(vector)

    assert c_m == v_c, f"Addition cannot be performed on a matrix and vector with size: ({r_m},{c_m}), (1, {v_c})"

    addition = [[0 for _ in range(r_m)] for _ in range(c_m)]

    for row in range(r_m):
        for column in range(c_m):
            addition[row][column] = matrix[row][column] + vector[column]

    return addition


def main() -> None:
    inputs: Matrix = [[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]]
    hidden_layer_weights: Matrix = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]
    hidden_layer_biases: Vector = [2, 3, 0.5]

    output_layer_weights: Matrix = [[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]]
    output_layer_biases: Vector = [-1, 2, -0.5]

    hidden_layer_output: Matrix = add_vector(dot(inputs, transpose(hidden_layer_weights)), hidden_layer_biases)
    output_layer_output: Matrix = add_vector(dot(hidden_layer_output, transpose(output_layer_weights)), output_layer_biases)

    print(output_layer_output)


if __name__ == "__main__":
    main()

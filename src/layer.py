# Implementation of a single layer consisting of 3 neurons

def main() -> None:
    inputs: list[float] = [1, 2, 3, 2.5]
    weights: list[list[float]] = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26 , -0.27 , 0.17 , 0.87]
    ]
    biases: list[float] = [2, 3, 0.5]

    output: list[float] = [
        sum([i * w for i, w in zip(inputs, weights[0])]) + biases[0],
        sum([i * w for i, w in zip(inputs, weights[1])]) + biases[1],
        sum([i * w for i, w in zip(inputs, weights[2])]) + biases[2],
    ]

    print(output)

if __name__ == '__main__':
    main()
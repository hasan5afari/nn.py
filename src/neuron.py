# Implementation of a single neuron with 4 inputs and bias


def main() -> None:
    inputs: list[float] = [1, 2, 3, 2.5]
    weights: list[float] = [0.2, 0.8, -0.5, 1.0]
    bias: float = 2.0

    output: float = sum([i * w for i, w in zip(inputs, weights)]) + bias

    print(output)


if __name__ == '__main__':
    main()
# Implementation of a single layer consisting of 3 neurons


def main() -> None:
    inputs: list[float] = [1, 2, 3, 2.5]
    weights: list[list[float]] = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26 , -0.27 , 0.17 , 0.87]
    ]
    biases: list[float] = [2, 3, 0.5]

    layer_outputs: list[float] = []

    for neuron_weights, neuron_bias in zip(weights, biases):
        layer_outputs.append(sum([i * w for i, w in zip(inputs, neuron_weights)]) + neuron_bias)

    print(layer_outputs)


if __name__ == '__main__':
    main()
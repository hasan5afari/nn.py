import numpy as np
import matplotlib.pyplot as plt

from dataset import (
    generate_sin_dataset,
    generate_spiral_dataset,
    generate_vertical_dataset,
)
from layers.dense_layer import DenseLayer
from activation_functions.relu import ReLU
from activation_functions.softmax import Softmax
from loss_functions.categorical_cross_entropy import CategoricalCrossEntropy


def main() -> None:
    np.random.seed(0)

    X, y = generate_spiral_dataset(2, 100)

    relu = ReLU()
    softmax = Softmax()
    categorical_cross_entropy = CategoricalCrossEntropy()

    hidden_layer = DenseLayer(2, 3)
    output_layer = DenseLayer(3, 2)

    hidden_layer.forward(X)
    hidden_layer_output = relu.forward(hidden_layer.get_output())

    output_layer.forward(hidden_layer_output)
    output_layer_output = softmax.forward(output_layer.get_output())

    loss = categorical_cross_entropy.calculate(output_layer_output, y)

    print(output_layer_output[:5])
    print(f"loss: {loss}")


if __name__ == "__main__":
    main()

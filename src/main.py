import numpy as np
import matplotlib.pyplot as plt

from dataset import (
    generate_sin_dataset,
    generate_spiral_dataset,
    generate_vertical_dataset
)
from layers.dense_layer import DenseLayer
from activation_functions.relu import ReLU


def main() -> None:
    np.random.seed(0)

    X, y = generate_spiral_dataset(3, 100)

    dl = DenseLayer(2, 3)
    dl.forward(X)

    print(dl.get_output()[:3])

    relu = ReLU()
    print(relu.forward(dl.get_output())[:3])


if __name__ == '__main__':
    main()
import numpy as np

from dataset import (
    generate_sin_dataset,
    generate_spiral_dataset,
    generate_vertical_dataset
)
from layers.dense_layer import DenseLayer


def main() -> None:
    np.random.seed(0)

    X, y = generate_spiral_dataset(3, 100)

    dl = DenseLayer(2, 3)
    dl.forward(X)

    print(dl.get_output()[:5])


if __name__ == '__main__':
    main()
import numpy as np


def initialize(random_seed: int = 42) -> None:
    np.random.seed(random_seed)


def generate_vertical_dataset(
    classes: int = 2, data_per_class: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((classes * data_per_class, 2))
    y = np.zeros(classes * data_per_class, dtype="uint8")

    for cls in range(classes):
        ix = range(classes * data_per_class, (cls + 1) * data_per_class)
        X[ix] = np.c_[
            np.random.randn(data_per_class) * 0.1 + (cls / 3),
            np.random.randn(data_per_class) * 0.1 + 0.5,
        ]
        y[ix] = cls

    return X, y


def generate_spiral_dataset(
    classes: int = 2, data_per_class: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((classes * data_per_class, 2))
    y = np.zeros(classes * data_per_class, dtype="uint8")

    for cls in range(classes):
        ix = range(classes * data_per_class, (cls + 1) * data_per_class)
        r = np.linspace(0.0, 1, data_per_class)
        t = (
            np.linspace(cls * 4, (cls + 1) * 4, data_per_class)
            + np.random.randn(data_per_class) * 0.2
        )
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = cls

    return X, y


def generate_sin_dataset(datapoints: int = 100) -> tuple[np.ndarray, np.ndarray]:
    X = np.arange(datapoints).reshape(-1, 1) / datapoints
    y = np.sin(2 * np.pi * X).reshape(-1, 1)

    return X, y

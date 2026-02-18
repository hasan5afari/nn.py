import numpy as np


class Accuracy:
    def calculate(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        if y.ndim == 2:
            y = np.argmax(y, axis=1)

        y_hat = np.argmax(y_hat, axis=1)

        return np.mean(y==y_hat)

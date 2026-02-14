import numpy as np

from .loss import LossFunction


class CategoricalCrossEntropy(LossFunction):
    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        samples = y_hat.shape[0]
        y_hat_clipped = np.clip(y_hat, 1e-7, 1 - 1e-7)

        if y.ndim == 1:
            correct_confidence = y_hat_clipped[np.arange(samples), y]
        elif y.ndim == 2:
            correct_confidence = np.sum(y_hat_clipped * y, axis=1)
        else:
            raise ValueError("y must be 1D or one-hot encoded")

        negative_log = -np.log(correct_confidence)

        return negative_log
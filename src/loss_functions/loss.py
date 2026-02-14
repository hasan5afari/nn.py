import numpy as np


class LossFunction:
    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        pass

    def calculate(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        samples_loss = self.forward(y_hat, y)
        average_loss = np.mean(samples_loss)

        return average_loss
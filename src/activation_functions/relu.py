import numpy as np


class ReLU:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.maximum(0, inputs)
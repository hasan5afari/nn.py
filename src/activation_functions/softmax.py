import numpy as np


class Softmax:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        exponential_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        normalized_probabilities = exponential_values / np.sum(exponential_values, axis=1, keepdims=True)

        return normalized_probabilities
import numpy as np


class DenseLayer:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self._weights = np.random.rand(n_neurons, n_inputs) * 0.01
        self._biases = np.zeros((1, n_neurons))
        self._output = np.zeros((1, n_neurons))


    def forward(self, inputs: np.array) -> None:
        self._output = np.dot(inputs, self._weights.T) + self._biases

    
    def get_output(self) -> np.array:
        return self._output
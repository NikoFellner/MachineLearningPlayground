import numpy as np
from Layers.Base import BaseLayer


class Sigmoid(BaseLayer):

    def __init__(self):
        BaseLayer.__init__(self)
        self.trainable = False
        self.activations = None

    def forward(self, input_tensor):
        # f(x) = 1 / (1+e^(x))
        next_input_tensor = np.divide(1, np.add(1, np.exp(np.multiply(-1, input_tensor))))
        self.activations = next_input_tensor
        return next_input_tensor

    def backward(self, error_tensor):
        # f'(x) = f(x) * (1 - f(x))
        derivative = np.multiply(self.activations, np.subtract(1, self.activations))
        next_error_tensor = np.multiply(derivative, error_tensor)
        return next_error_tensor

    def get_activation(self):
        return self.activations

    def set_activation(self, activation):
        self.activations = activation
        return
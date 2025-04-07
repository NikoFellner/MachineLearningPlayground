import numpy as np
from Layers.Base import BaseLayer


class TanH(BaseLayer):

    def __init__(self):
        BaseLayer.__init__(self)
        self.trainable = False
        self.activations = None

    def forward(self, input_tensor):
        # f(x) = tanh(x)
        next_input_tensor = np.tanh(input_tensor)
        self.activations = next_input_tensor
        return next_input_tensor

    def backward(self, error_tensor):
        # f'(x) = 1 - tanh(x)²
        tanh_square = np.square(self.activations)
        derivative = np.subtract(1, tanh_square)
        next_error_tensor = error_tensor * derivative
        return next_error_tensor

    def get_activation(self):
        return self.activations

    def set_activation(self, activation):
        self.activations = activation
        return
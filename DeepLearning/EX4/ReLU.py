import numpy as np
from Layers.Base import BaseLayer


class ReLU(BaseLayer):

    def __init__(self):
        BaseLayer.__init__(self)
        # setting the trainable value to False
        self.trainable = False
        self.input = ''

    def forward(self, input_tensor):
        # ReLU function in the forward pass is given by the mathematical expression f(x) = max{0.0,x}
        next_input_tensor = np.maximum(0.0, input_tensor)
        self.input = input_tensor
        return next_input_tensor

    def backward(self, error_tensor):
        # build the gradient with respect to the input, correlating the rule:
        # e_n-1 = 0 if x <= 0
        # and
        # e_n-1 = e_n for x > 0
        derivative = np.greater(self.input, 0.0) * 1
        # multiply element-wise with '*' operator and get the next error tensor
        next_error_tensor = error_tensor * derivative
        return next_error_tensor

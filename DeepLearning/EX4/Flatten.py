import numpy as np
from Layers.Base import BaseLayer


class Flatten(BaseLayer):

    def __init__(self):
        BaseLayer.__init__(self)
        self.trainable = False
        self.input_shape = None


    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        vector_length = 1

        batch_size = self.input_shape[0]
        sample_size = self.input_shape[1::]

        for dimension in sample_size:
            vector_length *= dimension

        reshaped_input_tensor = np.reshape(input_tensor, newshape=(batch_size, vector_length))
        return reshaped_input_tensor

    def backward(self, error_tensor):
        reshaped_error_tensor = np.reshape(error_tensor, newshape=self.input_shape)
        return reshaped_error_tensor
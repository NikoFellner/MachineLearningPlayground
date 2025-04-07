import numpy as np


class Flatten:

    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_shape = None


    def forward(self, input_tensor):
        # get the input shape and safe it global, also create a helper variable for the vector length
        self.input_shape = input_tensor.shape
        vector_length = 1

        # get the batch size and also the dimension of the sample
        batch_size = self.input_shape[0]
        sample_size = self.input_shape[1::]

        # get the needed vector length by iteration through the inputs and multiply the needed values
        for dimension in sample_size:
            vector_length *= dimension

        # simply reshape the tensor using the calculated vector length and the batch size
        reshaped_input_tensor = np.reshape(input_tensor, newshape=(batch_size, vector_length))
        return reshaped_input_tensor

    def backward(self, error_tensor):
        # reshape the error tensor to get a array by dimensions like the original input tensor
        reshaped_error_tensor = np.reshape(error_tensor, newshape=self.input_shape)
        return reshaped_error_tensor
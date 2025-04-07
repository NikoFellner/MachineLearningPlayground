import numpy as np


class SoftMax():

    def __init__(self):
        super().__init__()
        self.trainable = False
        self.next_input = ''

    def forward(self, input_tensor):
        # change the input tensor and decrease the possible outcome, that it doesn't calculate exploding outputs
        # that will appear if x_k getting larger and will further calculated with exp(x_k)
        # x'_k = x_k - max(x)
        opt_input_tensor = np.subtract(input_tensor, np.amax(input_tensor))
        # Softmax Function, the output will be calculated by a single input value (exponential), divided by the sum
        # of all inputs (exponential) in a batch
        # y_k = exp(x_k)/∑exp(x_j)
        next_input_tensor = np.exp(opt_input_tensor)/(np.sum(np.exp(opt_input_tensor), axis=1, keepdims=True))
        # next_input_tensor = np.transpose(next_input_tensor)
        self.next_input = next_input_tensor
        return next_input_tensor

    def backward(self, error_tensor):
        # Calculating element-wise per Batch (axis=0 in the sum)
        # the previous layer is calculated by the prediction, multiplied with the
        # subtraction of the error tensor and the sum of the error and prediction values in a batch
        # E_n-1 = y^ * (E_n - ∑E_n,j*y^_j)
        pred = np.transpose(self.next_input)
        error_tensor_transpose = np.transpose(error_tensor)
        next_error_tensor = np.multiply(pred, np.subtract(error_tensor_transpose, np.sum(np.multiply(error_tensor_transpose, pred), axis=0)))
        next_error_tensor = np.transpose(next_error_tensor)

        return next_error_tensor
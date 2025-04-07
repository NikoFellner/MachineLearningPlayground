import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = ''

    def forward(self, prediction_tensor, label_tensor):
        # creating the smallest possible epsilon = 2.220446049250313e-16
        epsilon = np.finfo(dtype=float).eps
        # calculating scheme for the loss:
        # y_max = np.multiply(prediction_tensor, label_tensor)
        # y_max2 = np.amax(np.transpose(y_max), axis=0)
        # y_max2_sum = np.add(y_max2, epsilon)
        # y_max_sum_log = np.log(y_max2_sum)
        # y_max_sum_log_minus = np.multiply(-1, y_max_sum_log)
        # y_max_sum_log_minus_sum = np.sum(y_max_sum_log_minus) = loss
        # formula: loss = ∑-ln(y^_k + ε) where y_k = 1
        # y_k is the label_tensor, if we multiply with the prediction-tensor, all other elements get zero.
        # after that, we just need to sum all max. values in one batch together
        loss = np.sum(np.multiply(-1, np.log(np.add(np.amax(np.transpose(np.multiply(prediction_tensor, label_tensor)), axis=0), epsilon))))
        self.prediction_tensor = prediction_tensor
        return loss

    def backward(self, label_tensor):
        # the previous error tensor gets calculated by the prediction and the labels
        # E_n = -(y/y^)
        prev_error_tensor = np.multiply(-1, (np.divide(label_tensor, self.prediction_tensor)))

        return prev_error_tensor









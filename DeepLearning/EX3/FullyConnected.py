import numpy as np


class FullyConnected:

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True

        # generate a input_tensor with bias (already transposed ones) variable, also safe the weights with bias
        # (already transposed) inside weights_variable, shape[n+1, m]
        self.input_tensor = ''
        self.weights = np.random.uniform(0, 1, size=((self.input_size + 1), self.output_size))
        # creating a optimizer variable
        self.optimizer = ''

    def forward(self, input_tensor):
        # getting the shape of the input tensor, which has "columns = input_size" and "rows = batch_size"
        # first index (shape[0]) are the rows, so the batch_size --> 'b'
        # second index (shape[1]) are the columns, so the input_size --> 'n'
        shape = input_tensor.shape
        # creating the input tensor with bias, shape(b, n+1)
        # input tensor has shape[b,n] when it gets inside the forward method
        # first create a bias with a row full of 1 and add to the n-size
        # safe it public afterwards
        input_bias = np.ones((shape[0], 1))
        input_tensor_with_bias = np.concatenate((input_tensor, input_bias), axis=1)
        self.input_tensor = input_tensor_with_bias

        # pushing forward to the next layer, so the inputs get multiplied with the corresponding weights
        # X' * W' = Y_^' --> X' = X_(Transpose), W' = W_(Transpose)...
        transpose_output = np.matmul(self.input_tensor, self.weights)
        # output tensor should be the input for the next layer
        return transpose_output

    def setOptimizer(self, optimizer):
        # setter, sets the Optimizer.Sgd class inside the _optimizer value
        self.optimizer = optimizer

    def getOptimizer(self):
        # getter method of the optimizer
        return self.optimizer

    def set_gradient_weights(self, gradient_weights):
        # The gradient with respect to the weights is formulated by the learning rate, the current error Tensor and the
        # transposed Input matrix
        # := E_(n) * X_(T)
        self.gradient_weights = gradient_weights

    def get_gradient_weights(self):
        # getter method of the gradient with respect to the weights
        return self.gradient_weights

    def backward(self, error_tensor):
        # the previous error tensor is defined by the weights tensor transposed multiplied by the error Tensor of the
        # current step (so the input error tensor) but excluding the bias
        # E_(n-1)_(T) = E_(n)_(T) * W_(T)_(T)
        # cutting of the bias
        weight_tensor_without_bias = np.delete(self.weights, -1, axis=0)
        transpose_weight_tensor = np.transpose(weight_tensor_without_bias)
        prev_error_tensor = np.matmul(error_tensor, transpose_weight_tensor)

        # update the weights with calculating the gradient with respect to the weights
        # W_(t+1)_(T) = W_(t)_(T) - η * E_(n)_(T) * X_(T)_(T) with δL/δW = E_(n)_(T) * X_(T)_(T)
        transpose_input_tensor = np.transpose(self.input_tensor)
        gradient_respect_weight = np.matmul(transpose_input_tensor, error_tensor)
        # doing the property thing with the gradient
        self.set_gradient_weights(gradient_respect_weight)

        # doing the weight optimization if layer has a optimizer
        if self.optimizer != '':
            updated_weights = self.optimizer.calculate_update(weight_tensor=self.weights,
                                                              gradient_tensor=gradient_respect_weight)
            self.weights = updated_weights
        return prev_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        reinitialized_weights = weights_initializer.initialize((self.input_size,self.output_size),
                                                               fan_in=self.input_size, fan_out=self.output_size )
        reinitialized_bias = bias_initializer.initialize(weights_shape=(1,self.output_size),
                                                               fan_in=1, fan_out=self.output_size)
        weights = np.concatenate((reinitialized_weights, reinitialized_bias), axis=0)
        self.weights = weights
        return
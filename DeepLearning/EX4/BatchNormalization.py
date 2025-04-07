import copy
import numpy as np
from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients
import Layers.Initializers


class BatchNormalization(BaseLayer):

    def __init__(self, channels):
        BaseLayer.__init__(self)
        self.channels = channels
        self.trainable = True

        self.decay = 0
        self.one_minus_decay = 1 - self.decay
        self.mean_value = None
        self.variance = None

        # weights and bias initialization
        self.weights = None
        self.gradient_weights = None
        self.bias = None
        self.gradient_bias = None
        self.initialize(weights_initializer=None, bias_initializer=None)

        self.input_shape = None
        self.input = None
        self.normalized_input = None

        self.optimizer = None

    def forward(self, input_tensor):
        epsilon = np.finfo(dtype=float).eps
        self.input_shape = input_tensor.shape
        self.input = input_tensor

        # testing if the input tensor does have the right shape, if it is the 4D, turn the tensor in some vector input,
        # therefore use the reformat function
        if len(input_tensor.shape) > 2:
            input_tensor = self.reformat(input_tensor)

        # test if the test or training is running
        if self.testing_phase == False:
            # if training is running follow the scheme below
            # calculate the mean value of the batch and also the variance of the batch
            # (variance = standard deviation to the power of two)
            # size of the mean- and variance-vector is equal to the channel dimension of the input tensor
            mean_value_batch = np.mean(input_tensor, axis=0, keepdims=True)     # = (µ_B)
            variance_batch = np.var(input_tensor, axis=0, keepdims=True)        # = (σ^2_B)

            # consider in further calculation the moving average of the mean (µ) and the variance  (σ^2)
            # µ˜_(k) = α * µ˜_(k-1) + (1 - α)* µ_B_(k)
            # σ^2˜_(k) = α * σ^2˜_(k - 1) + (1 - α) * σ^2_B_(k)
            # safe the values global
            if np.any(self.mean_value) == None:
                # for the first training set
                # µ˜_(k) = (1 - α)* µ_B_(k)
                # σ^2˜_(k) = (1 - α) * σ^2_B_(k)
                self.mean_value = np.multiply(self.one_minus_decay, mean_value_batch)
                self.variance = np.multiply(self.one_minus_decay, variance_batch)

            else:
                # until the second training set
                # µ˜_(k) = α * µ˜_(k-1) + (1 - α)* µ_B_(k)
                # σ^2˜_(k) = α * σ^2˜_(k - 1) + (1 - α) * σ^2_B_(k)
                tilde_mean_value = np.add(np.multiply(self.decay, self.mean_value),
                                          np.multiply(self.one_minus_decay, mean_value_batch))
                tilde_variance = np.add(np.multiply(self.decay, self.variance),
                                        np.multiply(self.one_minus_decay, variance_batch))

                self.mean_value = tilde_mean_value
                self.variance = tilde_variance

            # calculate the normalized input
            # X˜ = (X - µ_B) / √(σ^2_B_(k) + ε)
            numerator = np.subtract(input_tensor, mean_value_batch)     # = (X - µ_B)
            denominator = np.sqrt(np.add(variance_batch, epsilon))      # = √(σ^2_B_(k) + ε)
            normalized_input = np.divide(numerator, denominator)        # = X˜

        else:
            # same calculation in case of testing set, values just differentiate
            # using the updated values during training for the mean and the variance
            numerator = np.subtract(input_tensor, self.mean_value)
            denominator = np.sqrt(np.add(self.variance, epsilon))
            normalized_input = np.divide(numerator, denominator)

        self.normalized_input = normalized_input

        # Calculate the output tensor using the weights, bias and the input tensor
        # Y˜ = γ * X˜ + β
        # γ : weights
        # β : bias
        output = np.add(np.multiply(self.weights, normalized_input), self.bias)

        # reshape the tensor, in case, it was 4D at the entry of the forward pass
        if len(self.input_shape) > 2:
            output = self.reformat(output)

        return output

    def backward(self, error_tensor):
        error_tensor_shape = error_tensor.shape
        input_tensor = self.input

        # reformat the input- and the error tensor, if needed
        if len(error_tensor_shape) > 2:
            error_tensor = self.reformat(error_tensor)
            input_tensor = self.reformat(input_tensor)

        # gradient w.r.t. bias
        # simply the sum of the errors along the batch
        # ∑ E_b
        self.gradient_bias = np.sum(error_tensor, axis=0)

        # gradient w.r.t. weights, simply the sum over the batch after elementwise multiplication of
        # normalized input and error tensor
        # ∑ E_b * X˜_b
        self.gradient_weights = np.sum(np.multiply(error_tensor, self.normalized_input),axis=0)

        # gradient w.r.t. input
        # using provided code
        next_error_tensor = compute_bn_gradients(error_tensor=error_tensor, input_tensor=input_tensor, weights=self.weights,
                                                 mean=self.mean_value, var=self.variance)

        # if optimizer ist set, update bias and weights accordingly with calculate update method of the optimizer
        if self.optimizer != None:
            updated_weights = self.optimizer.calculate_update(weight_tensor=self.weights, gradient_tensor=self.gradient_weights)
            updated_bias = self.optimizer.calculate_update(weight_tensor=self.bias, gradient_tensor=self.gradient_bias)
            self.weights = updated_weights
            self.bias = updated_bias

        # reformat the next error tensor if need by using the reformat method
        if len(error_tensor_shape) > 2:
            next_error_tensor = self.reformat(next_error_tensor)
        return next_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        # weights getting a one initialization
        # bias getting zero initialization
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
        return

    def reformat(self, tensor):
        tensor_shape = tensor.shape
        # vector to image
        # accordingly to scheme below, just reverse
        if len(tensor_shape) == 2:
            shape_reshape_one = (self.input_shape[0], self.input_shape[3]* self.input_shape[2], self.input_shape[1])
            reshape_one = np.reshape(tensor, newshape=(shape_reshape_one))
            transposed_tensor = reshape_one.transpose((0, 2, 1))
            output_tensor = np.reshape(transposed_tensor, newshape=self.input_shape)

        # image to vector
        # step 1: reshape from (BxHxMxN) to (BxHxM*N)
        # step 2: transpose to get (BxM*NxH)
        # step 3: reshape to get (B*M*NxH)
        else:
            if len(tensor_shape) < 4:
                output_shape = (tensor_shape[0] * tensor_shape[2], tensor_shape[1])
                transposed_tensor = tensor.transpose((0,2,1))

            else:
                output_shape = (tensor_shape[0] * tensor_shape[2] * tensor_shape[3], tensor_shape[1])
                shape_reshape_one = (tensor_shape[0],tensor_shape[1], tensor_shape[2] * tensor_shape[3])
                reshape_one = np.reshape(tensor, newshape=(shape_reshape_one))
                transposed_tensor = reshape_one.transpose((0, 2, 1))

            output_tensor = np.reshape(transposed_tensor, newshape=(output_shape))

        return output_tensor

    def setOptimizer(self, optimizer):
        self.setOptimizerWeights(copy.deepcopy(self.optimizer))
        self.setOptimizerBias(copy.deepcopy(self.optimizer))
        return

    def setOptimizerWeights(self, optimizer):
        self.optimizer_weights = optimizer
        return

    def setOptimizerBias(self, optimizer):
        self.optimizer_bias = optimizer
        return

    def calculate_regularization_loss(self):
        regularization_loss_weights = self.optimizer.regularizer.norm(self.weights)
        regularization_loss_bias = self.optimizer.regularizer.norm(self.bias)
        regularization_loss = regularization_loss_bias + regularization_loss_weights
        return regularization_loss
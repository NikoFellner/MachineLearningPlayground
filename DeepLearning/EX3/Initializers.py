# fully connected layers
# fan_in : input dimension of weights
# fan_out : output dimension of weights

# convolutional layers
# fan_in : [input channels x kernel height x kernel width]
# fan_out : [output channels x kernel height x kernel width]
import numpy as np


class Constant:
    def __init__(self, constant_value=0.1):
        self.constant_value = constant_value

    def initialize(self, weights_shape, fan_in, fan_out):
        weights_initialized = np.multiply(np.ones(weights_shape), self.constant_value)
        return weights_initialized


class UniformRandom:

    def initialize(self, weights_shape, fan_in, fan_out):
        weights_initialized = np.random.uniform(0, 1, size=weights_shape)
        return weights_initialized


class Xavier:

    def initialize(self, weights_shape, fan_in, fan_out):
        # initialize the weights with a zero mean gaussian
        # (mean: centre of the distribution)
        # (scale: standard deviation here σ = √(2/(fan_in+fan_out))
        mean = 0
        standard_deviation = np.sqrt(2/(fan_in+fan_out))
        weights_initialized = np.random.normal(loc=mean, scale=standard_deviation, size=weights_shape)
        return weights_initialized


class He:

    def initialize(self, weights_shape, fan_in, fan_out):
        # initialize the weights with a zero mean gaussian
        # (mean: centre of the distribution)
        # (scale: standard deviation here σ = √(2/fan_in)
        mean = 0
        standard_deviation = np.sqrt(2 /fan_in)
        weights_initialized = np.random.normal(loc=mean, scale=standard_deviation, size=weights_shape)
        return weights_initialized
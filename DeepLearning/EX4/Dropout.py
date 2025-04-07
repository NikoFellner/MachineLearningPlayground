import numpy as np
from Layers.Base import BaseLayer

class Dropout(BaseLayer):

    def __init__(self, probability):
        BaseLayer.__init__(self)
        self.probability = probability
        self.mask = None


    def forward(self, input_tensor):
        input_shape = input_tensor.shape
        # differentiate testing and training phase
        # drop nodes out with the drop-probability = 1 - probability
        # therefore use a mask, where the randomly dropped nodes are set and saved global
        # no dropout in testing phase, in training multiply the input with 1/probability and with the created mask,
        # to dropout the nodes
        if self.testing_phase == True:
            drop_probability = 1 - self.probability
            self.mask = np.random.uniform(0, 1, input_shape) > drop_probability
            output = input_tensor
        else:
             drop_probability = 1 - self.probability
             self.mask = np.random.uniform(0, 1, input_shape) > drop_probability
             activations_multiplied = np.multiply(input_tensor, 1/self.probability)
             output = np.multiply(activations_multiplied, self.mask)
        return output

    def backward(self, error_tensor):
        # drop the corresponding errors out, with the mask created in the forward pass
        # and also multiply the errors by 1/probability
        gradient = np.multiply(error_tensor, self.mask)
        output = np.multiply(gradient, 1/self.probability)
        return output
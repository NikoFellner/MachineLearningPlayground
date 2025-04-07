import copy
from Layers.Base import BaseLayer

from Layers.FullyConnected import FullyConnected
import numpy as np

class NeuralNetwork(BaseLayer):

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        BaseLayer.__init__(self)
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = ''
        self.loss_layer = ''
        self.input_data = ''

        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.regularization_loss = None

    def forward(self):
        self.regularization_loss = 0
        # self.input_data, input-tensor and label-tensor
        # input_data : self.input_data[0]; label_tensor : self.input_data[1]
        self.input_data = self.data_layer.next()
        input_tensor = self.input_data[0]

        # forward method of each layer needs the input_tensor
        # the return of the forward of each layer is the input for the next one
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            if (layer.trainable == True):
                if (layer.optimizer != None):
                    if(layer.optimizer.regularizer != None):
                        self.regularization_loss += layer.calculate_regularization_loss()

        # creating the loss after the last layer and return it
        data_return = self.loss_layer.forward(prediction_tensor=input_tensor, label_tensor=self.input_data[1])

        return data_return

    def backward(self):
        # create the previous error tensor using the loss_layer and the used label_tensor
        prev_error_tensor = self.loss_layer.backward(self.input_data[1])

        # going through the Neural Network backwards using [::-1]
        for layer in self.layers[::-1]:
            prev_error_tensor = layer.backward(prev_error_tensor)

        # return the last previous error tensor of the backward pass
        return prev_error_tensor

    def append_layer(self, layer):
        # if we have a trainable layer and it is the first trainable layer, then set the optimizer for this layer
        # if there is already a trainable layer inside the network, we do not want to set the optimizer again
        # therefore we used a variable, which gets True, if we already have a trainable layer
        if (layer.trainable == True):
            copy_optimizer = copy.deepcopy(self.optimizer)
            layer.setOptimizer(copy_optimizer)

            copy_weights_init = copy.deepcopy(self.weights_initializer)
            copy_weights_bias_init = copy.deepcopy(self.bias_initializer)
            layer.initialize(copy_weights_init, copy_weights_bias_init)

        # add the layer to the layers-list, independent if it is trainable or not
        self.layers.append(layer)
        return

    def train(self, iterations):
        # train the network with a given number of iteration, after every forward pass, we save the loss of the
        # iteration inside the loss-list
        # go the network backwards and start again with the forward pass
        for _ in range(iterations):
            loss = self.forward()
            if self.regularization_loss != 0:
                loss = loss + self.regularization_loss
            self.backward()
            self.loss.append(loss)
        return

    def test(self, input_tensor):
        # get the output of the last layer, but not the loss-layer and return the tensor
        for layer in self.layers:
            layer.testing_phase = True
            input_tensor = layer.forward(input_tensor)
        return input_tensor

    def set_phase(self, phase, layer):
        layer.testing_phase = phase
        return

    def get_phase(self, layer):
        return layer.testing_phase

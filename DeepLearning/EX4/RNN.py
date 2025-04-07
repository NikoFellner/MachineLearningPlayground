import numpy as np
import scipy.signal

from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid


class RNN(BaseLayer):

    def __init__(self, input_size, hidden_size, output_size):
        BaseLayer.__init__(self)
        self.tanh = TanH()
        self.sigmoid = Sigmoid()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.memorize = False
        self.last_hidden_vector = np.zeros(shape=(1, self.hidden_size))
        self.trainable = True
        # dimension of the hidden vector (1, self.hidden_size)
        self.first_hidden_vector = np.zeros(shape=(1, self.hidden_size))
        self.hidden_state = None
        # dimension of the bias for the fc1 (1, self.hidden_size)
        self.bias_fc1 = np.ones((1, 1))
        # dimension of the bias for the fc2 (1, self.output_size)
        self.bias_fc2 = np.ones((1, 1))
        # dimension of the input for the first fully connected layer: input_vector + hidden_vector + bias_vector
        # output dimension is the size of the hidden vector
        self.fc1 = FullyConnected(input_size=(self.input_size+self.hidden_size), output_size=self.hidden_size)
        self.fc1.trainable = False
        # dimension of the input for the second fully connected layer: hidden_vector + output_vector
        # output dimension is the size of the output vector
        self.fc2 = FullyConnected(input_size=(self.hidden_size), output_size=self.output_size)
        self.fc2.trainable = False

        self.gradient_hiddenstate = None
        self.tanh_activations = None
        self.sigmoid_activations = None
        self.gradient_weights_fc1 = None
        self.gradient_weights_fc2 = None
        self.output_forward = None
        self.first_call = True
        self.input_tensor_fc1 = None
        self.input_tensor_fc2 = None

        self.weights = self.fc1.weights
        self.gradient_weights = None

        self.optimizer = None

    def forward(self, input_tensor):
        BATCH_SIZE = input_tensor.shape[0]

        self.tanh_activations = np.zeros(shape=(BATCH_SIZE, 1, self.hidden_size))
        self.sigmoid_activations = np.zeros(shape=(BATCH_SIZE, 1, self.output_size))
        self.input_tensor_fc1 = np.zeros(shape=(BATCH_SIZE, 1, self.input_size+self.hidden_size+1))
        self.input_tensor_fc2 = np.zeros(shape=(BATCH_SIZE, 1, self.hidden_size+1))


        # dummy arrays
        next_hidden_state = np.empty((BATCH_SIZE, self.hidden_size))
        output_yt = np.empty((BATCH_SIZE, self.output_size))
        input_concatenated_fc1 = np.empty((1, self.input_size + self.hidden_size))
        reformated_input_fc2 = np.empty((1, self.hidden_size))

        for timestep in range(BATCH_SIZE):
            # at initializing point, hidden state is zero valued
            # 1: concatenate input x_t and hidden state h_t plus a single 1, then we get x~_t
            # the input_size of the first FC is - input_size+hidden_size, output_size is hidden_size
            if timestep == 0:
                # if memorize is true, the last hidden vector is the last hidden hidden vector of the forward pass
                # in the last sequence, else it stays as zero like tensor
                input_concatenated_fc1[0] = np.concatenate((self.last_hidden_vector[0], input_tensor[timestep]), axis=0)
            else:
                # for any other timestep the hiddenstate of the previous timestep should be taken
                input_concatenated_fc1[0] = np.concatenate((next_hidden_state[timestep - 1], input_tensor[timestep]), axis=0)

            # 2: forward passing through the fc1 with the input_tensor = x~_t
            output_fc1 = self.fc1.forward(input_tensor=input_concatenated_fc1)
            # 2_1: save the created input needed for the corresponding fc1 backward pass
            self.input_tensor_fc1[timestep] = self.fc1.get_input()

            # 3: the output tensor of the fc1 should be passed through the forward method of the tanh
            output_tanh = self.tanh.forward(input_tensor=output_fc1)
            # 3_1: save the activation for the corresponding tanh backward pass
            self.tanh_activations[timestep] = self.tanh.get_activation()
            # 3_2: the output of the tanh is the next hidden state h_t+1
            next_hidden_state[timestep] = output_tanh

            # between 3_2 and 4 happens a copy procedure

            # 4: push the h_t+1 through the fc2 as input tensor
            reformated_input_fc2[0] = next_hidden_state[timestep]
            output_fc2 = self.fc2.forward(input_tensor=reformated_input_fc2)
            # 4_1: save the created input needed for the corresponding fc2 backward pass
            self.input_tensor_fc2[timestep] = self.fc2.get_input()

            # 5: the output tensor is the input for the sigmoid fct. pass the tensor through the forward method
            output_sigmoid = self.sigmoid.forward(input_tensor=output_fc2)
            # 5_1: save the activation fpr the corresponding sigmoid backward pass
            self.sigmoid_activations[timestep] = self.sigmoid.get_activation()

            # 6: the output is the returned tensor in the forward method of the elman cell
            output_yt[timestep] = output_sigmoid[0]

        self.output_forward = output_yt
        # save the hidden state local for the backward pass if needed
        self.hidden_state = next_hidden_state

        # if memorize is true, set the last hidden vector variable to the values of the last calculated hidden vector
        if self.memorize == True:
            self.last_hidden_vector[0] = next_hidden_state[-1]
        return output_yt

    def backward(self, error_tensor):
        BATCH_SIZE = error_tensor.shape[0]

        self.gradient_hiddenstate = np.zeros((1, self.hidden_size))
        splitted_gradient_fc2 = np.empty(shape=(1, self.hidden_size))
        reformated_split_hidden_state = np.empty(shape=(1, self.hidden_size))
        reformated_split_input = np.empty(shape=(1, self.input_size))
        gradient_input = np.empty(shape=(BATCH_SIZE, self.input_size))
        self.gradient_weights_fc1 = np.zeros(shape=(self.hidden_size+self.input_size+1,self.hidden_size))
        self.gradient_weights_fc2 = np.zeros(shape=(self.hidden_size+1, self.output_size))


        for timestep in range(BATCH_SIZE):
            timestep = BATCH_SIZE - timestep - 1

            # ------------------------- SIGMOID
            # getting through the elman cell in reverse
            # first backpropagate through the sigmoid fct. with the given error tensor
            self.sigmoid.set_activation(self.sigmoid_activations[timestep])  # correspond to the right stored activation
            gradient_sigmoid = self.sigmoid.backward(error_tensor[timestep])

            # ------------------------- FC2
            # with the calculated gradient, go into the fully connected fc2
            # self.fc2.input_tensor = self.input_tensor_fc2[timestep]
            self.fc2.set_input(self.input_tensor_fc2[timestep])
            gradient_fc2 = self.fc2.backward(gradient_sigmoid)
            actual_gradient_weights_fc2 = self.fc2.get_gradient_weights()
            self.gradient_weights_fc2 = np.add(self.gradient_weights_fc2, actual_gradient_weights_fc2)

            # --------------------- SPLIT FC2
            # need to split the gradient -> separate the bias and propagate along
            splitted = np.split(gradient_fc2[0], [self.hidden_size])
            splitted_gradient_fc2[0] = splitted[0]

            # backwardpass of a copy procedure is a sum
            # the backpropagated gradient for the tanh function is the sum of the gradient of h_t and gradient fc2
            # in the first iteration the h_t gradient is zero, so just the gradient of the fc2 is propagated
            gradient_copy2 = np.add(splitted_gradient_fc2, self.gradient_hiddenstate)

            # ----------------------- TANH
            # this gradient gets backpropagated through the tanh
            self.tanh.set_activation(self.tanh_activations[timestep])     # correspond to the right stored activation
            gradient_tanh = self.tanh.backward(gradient_copy2)

            # ----------------------- FC1
            # now backpropagate through the fc1
            self.fc1.set_input(self.input_tensor_fc1[timestep])
            gradient_fc1 = self.fc1.backward(gradient_tanh)
            actual_gradient_weights_fc1 = self.fc1.get_gradient_weights()
            self.gradient_weights_fc1 = np.add(self.gradient_weights_fc1, actual_gradient_weights_fc1)

            # ---------------------- SPLIT FC1
            # the result is gradient for x~, so for h_t-1 and also x_t and b
            # to get the gradient w.r.t. input, split the gradient of the fc1 accordingly
            # for all timesteps save the gradient corresponding to the input in a tensor and return it
            # split1 inherit the vector for the hidden vector on position 1 and the rest on positon 2
            split_fc1_1 = np.split(gradient_fc1[0], [self.hidden_size])
            # the gradient w.r.t. hidden state we get out the fc1 is now the summed part before backwarding the tanh fct.
            reformated_split_hidden_state[0] = split_fc1_1[0]
            self.gradient_hiddenstate = reformated_split_hidden_state
            # split2 inherit the vector for the input
            split_fc1_2 = np.split(split_fc1_1[1], [self.input_size])
            reformated_split_input[0] = split_fc1_2[0]
            gradient_input[timestep] = reformated_split_input[0]
        # do one optimization step for the fc1 and fc2 at the end of the backpropagation, sum over all gradients
        # for the timesteps
        self.gradient_weights = self.gradient_weights_fc1


        if self.optimizer != None:
            weights_fc1 = self.fc1.weights
            self.fc1_weights = self.optimizer.calculate_update(weight_tensor=weights_fc1,
                                            gradient_tensor=self.gradient_weights_fc1)
            self.fc1.weights = self.fc1_weights

            weights_fc2 = self.fc2.weights
            self.fc2.weights = self.optimizer.calculate_update(weight_tensor=weights_fc2,
                                            gradient_tensor=self.gradient_weights_fc2)

        return gradient_input

    def initialize(self, weights_initializer, bias_initializer):
        self.fc1.initialize(weights_initializer, bias_initializer)
        self.fc2.initialize(weights_initializer, bias_initializer)
        return

    def calculate_regularization_loss(self):
        regularization_loss_fc1 = self.fc1.calculate_regularization_loss()
        regularization_loss_fc2 = self.fc1.calculate_regularization_loss()
        regularization_loss = regularization_loss_fc1 + regularization_loss_fc2
        return regularization_loss

    def setOptimizer(self, optimizer):
        self.optimizer = optimizer
        self.fc1.setOptimizer(optimizer)
        self.fc2.setOptimizer(optimizer)
        return

    @property
    def weights(self):
       return self.fc1.weights

    @weights.setter
    def weights(self, x):
        self.fc1.weights = x



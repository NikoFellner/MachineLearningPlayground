import numpy as np


class Sgd:

    def __init__(self, learning_rate):
        self.learning_rate = float(learning_rate)

    def calculate_update(self, weight_tensor, gradient_tensor):
        # implementation of the Stochastic Gradient Descent Algorithm, simplified by using the weight tensor and also
        # the gradient tensor as input, the learning rate is set in the constructor - so it is:
        # w_(k+1) = w_(k) - η * ∇L(w_(k))
        # w : weights       η : learning rate       k : step notation       ∇L(w_(k)): Gradient
        weight_tensor_update = np.subtract(weight_tensor, np.multiply(self.learning_rate, gradient_tensor))
        return weight_tensor_update

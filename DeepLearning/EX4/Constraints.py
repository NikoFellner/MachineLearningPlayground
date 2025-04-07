import numpy as np


class L2_Regularizer:

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        # outer formula: (1- η * λ) *w_k − η * (∂L/∂w_k)
        # reformulate: w_k - η * λ *w_k − η * (∂L/∂w_k)
        # here to solve just the part (subgradient): λ * w_k
        sub_gradient_weights = np.multiply(weights, self.alpha)
        return sub_gradient_weights

    def norm(self, weights):
        # calculating the squared L2 norm of the weights with numpy fct. linalg.norm(x, ord, axis, keepdims)
        # x : Input array. If axis is None, x must be 1-D or 2-D, unless ord is None. If both axis and ord are None,
        # the 2-norm of x.ravel will be returned
        # ord : defines the norm of the matrix - None = Frobenius norm (2-norm)
        # axis : If axis is None then either a vector norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned
        # formula: λ||w||2_2
        norm = np.linalg.norm(weights)
        norm_squared = np.square(norm)
        norm_squared_alpha = norm_squared * self.alpha
        return norm_squared_alpha


class L1_Regularizer:

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        # formula: w_k - η * λ * sign(w_k) − η * (∂L/∂w_k)
        # here to solve just the part (subgradient): λ * sign(w_k)
        #sign_weights = np.sign(weights)
        sub_gradient_weights = np.multiply(self.alpha, np.sign(weights))
        return sub_gradient_weights

    def norm(self, weights):
        # formula: λ||w||1
        norm = np.linalg.norm(np.linalg.norm(weights, ord=1, axis=1), ord=1)
        norm_alpha = norm * self.alpha
        return norm_alpha

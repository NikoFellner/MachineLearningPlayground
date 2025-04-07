import numpy as np
import copy


class Optimizers:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer
        return


class Sgd(Optimizers):

    def __init__(self, learning_rate):
        Optimizers.__init__(self)
        self.learning_rate = float(learning_rate)

    def calculate_update(self, weight_tensor, gradient_tensor):
        # implementation of the Stochastic Gradient Descent Algorithm, simplified by using the weight tensor and also
        # the gradient tensor as input, the learning rate is set in the constructor - so it is:
        # w_(k+1) = w_(k) - η * ∇L(w_(k))
        # w : weights       η : learning rate       k : step notation       ∇L(w_(k)): Gradient
        if self.regularizer == None:
            weight_tensor_update = np.subtract(weight_tensor, np.multiply(self.learning_rate, gradient_tensor))
        else:
            # regularizer is set (e.g. L1)
            # formula: w_(k+1) = w_k - η * λ * sign(w_k) − η * (∂L / ∂w_k)
            # with shrinkage: w_k - η * λ * sign(w_k)
            # and subgradient: λ * sign(w_k)
            subgradient = self.regularizer.calculate_gradient(weight_tensor)
            subgradient_learning_rate = np.multiply(self.learning_rate, subgradient)
            shrinkage = np.subtract(weight_tensor, subgradient_learning_rate)
            weight_tensor_update = np.subtract(shrinkage, np.multiply(self.learning_rate, gradient_tensor))

        return weight_tensor_update


class SgdWithMomentum(Optimizers):

    def __init__(self, learning_rate, momentum_rate):
        Optimizers.__init__(self)
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.momentum_tensor = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        # if it is the first time the momentum tensor gets called, just multiply the gradient and the learning rate
        # after first iteration include the previous momentum tensor
        # v_(k) = µ*v_(k-1) - η * ∇L(w_(k))

        if np.any(self.momentum_tensor) == None:
            next_momentum_tensor = np.multiply(np.multiply(self.learning_rate, gradient_tensor),-1)
        else:
            next_momentum_tensor = np.subtract(np.multiply(self.momentum_rate, self.momentum_tensor),
                                   np.multiply(self.learning_rate, gradient_tensor))

        self.momentum_tensor = next_momentum_tensor

        if self.regularizer == None:
            # update the weights respecting the momentum
            # w_(k+1) = w_(k) + v_(k)
            updated_weights = np.add(weight_tensor, self.momentum_tensor)
        else:
            # regularizer is set (e.g. L1)
            # formula: w_(k+1) = w_k - η * λ * sign(w_k) + v_k
            # with shrinkage: w_k - η * λ * sign(w_k)
            # and subgradient: λ * sign(w_k)
            subgradient = self.regularizer.calculate_gradient(weight_tensor)
            subgradient_learning_rate = np.multiply(self.learning_rate, subgradient)
            shrinkage = np.subtract(weight_tensor, subgradient_learning_rate)
            updated_weights = np.add(shrinkage, self.momentum_tensor)

        return updated_weights


class Adam(Optimizers):

    def __init__(self, learning_rate, mu, rho):
        Optimizers.__init__(self)
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

        self.gradient = None
        self.momentum_tensor = None
        self.rho_tensor = None
        self.iteration_number = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        # increasing the iteration number
        self.iteration_number += 1

        # gradient tensor
        # g_(k) = ∇L(w_(k))
        self.gradient = gradient_tensor

        # Momentum Tensor
        # v_(k) = µ*v_(k-1) + (1-µ)*g_(k)
        if np.any(self.momentum_tensor) == None:
            momentum_tensor = np.multiply((1-self.mu), gradient_tensor)
        else:
            momentum_tensor = np.add(np.multiply(self.mu, self.momentum_tensor),
                                     np.multiply((1-self.mu), gradient_tensor))
        self.momentum_tensor = momentum_tensor
        # Bias correction for the Momentum Tensor
        # v^_(k) = v_(k) / (1-µ**k)
        corrected_momentum_tensor = np.divide(momentum_tensor, (1-self.mu**self.iteration_number))

        # rho Tensor
        # r_(k) = ρ*r_(k−1) + (1 − ρ)*g_(k)☉g_(k)
        if np.any(self.rho_tensor) == None:
            rho_tensor = np.multiply(np.multiply((1 - self.rho), gradient_tensor), gradient_tensor)
        else:
            rho_tensor = np.add(np.multiply(self.rho, self.rho_tensor),
                                np.multiply(np.multiply((1 - self.rho), gradient_tensor), gradient_tensor))

        self.rho_tensor = rho_tensor

        # Bias correction for the rho tensor
        # r^_(k) = r_(k) / (1-ρ**k)
        corrected_rho_tensor = np.divide(rho_tensor, (1-self.rho**self.iteration_number))

        # creating the smallest possible number, epsilon = 2.220446049250313e-16
        epsilon = np.finfo(dtype=float).eps

        # w_(k+1) = w_(k) - η * v^_(k)/(√r^_(k) + ε)
        # step1 = np.sqrt(corrected_rho_tensor)
        # step2 = np.add(step1, epsilon)
        # step3 = np.multiply(self.learning_rate, corrected_momentum_tensor)
        # step4 = np.divide(step3, step2)
        # step5 = np.subtract(weight_tensor, step4)
        if self.regularizer == None:
            updated_weights = np.subtract(weight_tensor, np.divide(np.multiply(self.learning_rate, corrected_momentum_tensor),
                                                               np.add(np.sqrt(corrected_rho_tensor), epsilon)))
        else:
            # regularizer is set (e.g. L1)
            # formula: w_(k+1) = w_k - η * λ * sign(w_k) + v_k
            # with shrinkage: w_k - η * λ * sign(w_k)
            # and subgradient: λ * sign(w_k)
            subgradient = self.regularizer.calculate_gradient(weight_tensor)
            subgradient_learning_rate = np.multiply(self.learning_rate, subgradient)
            shrinkage = np.subtract(weight_tensor, subgradient_learning_rate)
            updated_weights = np.subtract(shrinkage,
                                          np.divide(np.multiply(self.learning_rate, corrected_momentum_tensor),
                                                    np.add(np.sqrt(corrected_rho_tensor), epsilon)))
        return updated_weights

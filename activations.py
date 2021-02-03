import numpy as np


class Activation:
    def __call__(self, x):
        raise NotImplementedError

    def deriv(self, y, activated=True):
        """
        Return the derivative value of the activation.
        
        Args:
            y: the input value at which the derivative is calculated
            activated_input: whether the input y has been activated
        """
        raise NotImplementedError


class Tanh(Activation):
    def __call__(self, x):
        return np.tanh(x)

    def deriv(self, y, activated=True):
        if activated:
            return 1 - y**2
        else:
            return 1 - self(y)**2


class Logistic(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def deriv(self, y, activated=True):
        if activated:
            return y * (1 - y)
        else:
            return np.exp(y) / (1 + np.exp(y))**2


class ReLU(Activation):
    @staticmethod
    def __call__(x):
        return np.maximum(x, 0)

    @staticmethod
    def deriv(y, activated=True):
        return (y > 0).astype(float)

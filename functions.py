import numpy as np
from utils import sign, bernoulli, baseclass, abstractmethod


class Function(baseclass):
    """A function that supports both forward and backward passes."""
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)    
    
    @abstractmethod
    def forward(self, *args, **kwds):
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args, **kwds):
        raise NotImplementedError


class Dropout(Function):
    """Randonly deactivates some neurons to mitigate overfitting."""

    def __init__(self, p, size):
        self.p = p
        self.size = size
        self._mask = None

    def forward(self, input):
        mask = bernoulli(self.size, 1 - self.p) / (1 - self.p)
        self._mask = mask
        return mask * input

    def backward(self):
        return self._mask
    
    
class Loss(Function):
    """Base class of loss functions."""
    
    @classmethod
    def get(cls, loss):
        """Get the Loss instance corresponding to the argument."""
        if type(loss) is str:
            loss = loss.lower()
            if loss[0] == 'l' and loss[1:].isdigit():
                return L(int(loss[1:]))
            elif loss in ['crossentropy', 'cross_entropy', 'ce']:
                return CrossEntropy()
            elif loss in ['softmax_crossentropy', 'softmax_ce',
                          'softmax_cross_entropy', 'smce']:
                return SoftMaxCE()
            else:
                raise ValueError(f"unknown loss function: {loss}")
        elif isinstance(loss, cls):
            return loss
        else:
            raise ValueError(f"unknown loss function: {loss}")
    
    @abstractmethod
    def forward(self, output, target):
        """Compute the loss between the output and the target."""
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, output, target):
        """Compute the gradient of the loss w.r.t the output."""
        raise NotImplementedError


class L(Loss):
    """Loss function defined in the L^p space."""
    
    def __init__(self, p=2):
        """
        Args:
            p: the value of p in `L^p`
                If p = 2, the loss is the square sum of residuals;
                if p = 1, it is the sum of absolute residuals.
        """
        self.p = p
        
    def forward(self, output, target):
        """Compute the p-th power of the p-norm of the residuals."""
        res = output - target  # residuals
        loss = np.sum((np.abs(res) if self.p % 2 else res) ** self.p)
        return loss
    
    def backward(self, output, target):
        res = output - target
        if self.p % 2:
            return (sign(res) if self.p == 1 else
                    sign(res) * res ** (self.p - 1))
        else:
            return (res if self.p == 2 else
                    res ** (self.p - 1))


class CrossEntropy(Loss):
    """Cross entropy loss, usually used in classification."""
    
    def forward(self, output, target):
        return - target @ np.log(output)
    
    def backward(self, output, target):
        return - target / output
    
    
class SoftMaxCE(Loss):
    """Cross entropy with softmax transformation."""
    
    def forward(self, output, target):
        """The output and the target should be probability distributions."""
        exp_sum = np.sum(np.exp(output), axis=-1)
        dot_prod = np.sum(output * target, axis=-1)
        return np.sum(np.log(exp_sum) - dot_prod)
    
    def backward(self, output, target):
        return output - target


class Activation(Function):
    """A nonlinear activation function."""
    
    @abstractmethod
    def backward(self, y):
        "Note that the value passed backward should have been activated."
        raise NotImplementedError

    
class ActivationAccess:
    """Manages the activation attribute of a layer."""
    
    def __set__(self, layer, value):
        if type(value) is str:
            value = value.lower()
            if value == 'tanh':
                f = Tanh()
            elif value in ['logistic', 'sigmoid']:
                f = Logistic()
            elif value == 'relu':
                f = ReLU()
            elif value == 'linear':
                f = Linear()
            elif value == 'default':
                f = False
            else:
                raise ValueError(f"unknown activation function: {value}")
        elif isinstance(value, Activation):
            f = value
        elif not value:
            f = None
        else:
            raise ValueError(f"unknown activation function: {value}")
        layer._activation = f

    def __get__(self, layer, type=None):
        return layer._activation
    
    
class Linear(Activation):
    """Equivalent to no activation."""
    
    def __bool__(self):
        return False


class Tanh(Activation):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, y):
        return 1 - y**2


class Logistic(Activation):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, y):
        return y * (1 - y)


class ReLU(Activation):
    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, y):
        return (y > 0).astype(np.float)


class SoftMax(Activation):
    def forward(self, x):
        ex = np.exp(x)
        return ex / np.sum(ex, axis=-1)
    
    def backward(self, y):
        # TODO: fix this!
        return np.array([])

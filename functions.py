import numpy as np
from utils import sign, bernoulli, baseclass, abstractmethod


class Function(baseclass):
    """A function that supports both forward and backward passes."""
    
    def __call__(self, x):
        return self.forward(x)    
    
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def backward(self, y):
        raise NotImplementedError


class Dropout(Function):
    """Randonly deactivates some neurons to mitigate overfitting."""

    def __init__(self, p, size):
        self.p = p
        self.size = size
        self._mask = None

    def forward(self, input):
        mask = bernoulli(self.input_dim, 1 - self.p) / (1 - self.p)
        self._mask = mask
        return mask * input

    def backward(self, error):
        return error * self._mask
    
    
class Loss(Function):
    """Base class of loss functions."""
    
    @classmethod
    def get(cls, loss):
        if type(loss) is str:
            loss = loss.lower()
            if loss[0] == 'l' and loss[1:].isdigit():
                return L(int(loss[1:]))
            elif loss in ['crossentropy', 'cross_entropy', 'cross entropy']:
                return CrossEntropy()
            else:
                raise ValueError(f"unknown loss function: {loss}")
        elif isinstance(loss, cls):
            return loss
        else:
            raise ValueError(f"unknown loss function: {loss}")
    
    @abstractmethod
    def forward(self, output, target):
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, output, target):
        raise NotImplementedError
    
    
class L(Loss):
    """Standard loss function defined in the Lp space."""
    
    def __init__(self, p=2):
        """The p-th power of the p-norm of the residuals.
        If l = 2, it is the square sum of residuals;
        if l = 1, it is the sum of absolute residuals.
        """
        self.p = p
        self._res = None  # record the residuals during forward
        
    def forward(self, output, target, average=True):
        target = np.reshape(target, [len(target), -1])
        self._res = output - target
        
        loss = np.sum((np.abs(self._res) if self.l % 2
                       else self._res) ** self.p)
        
        if average: loss /= len(output)
        return loss
    
    def backward(self, output, target):
        if self.l % 2:
            return (sign(self._res) if self.l == 1 else
                    sign(self._res) * self._res ** (self.p - 1))
        else:
            return (self._res if self.l == 2 else
                    self._res ** (self.p - 1))
            

class CrossEntropy(Loss):
    """Cross entropy loss, usually used in classification."""


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


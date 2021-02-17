import numpy as np
from utils import sign, bernoulli, baseclass, abstractmethod, wraps


class Function(baseclass):
    """A function that supports both forward and backward passes."""
    
    def __init__(self):
        self.forward = self.record_result(self.forward)
        
    def record_result(self, f):
        _f = f  # store the original function
        @wraps(f)
        def call(*args, **kwds):
            self._ret = _f(*args, **kwds)
            return self._ret
        return call
    
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
        super().__init__()
        self.p = p
        self.size = size
        self._mask = None

    def forward(self, x):
        mask = bernoulli(self.size, 1 - self.p) / (1 - self.p)
        self._mask = mask
        return mask * x

    def backward(self, err):
        return self._mask * err
    
    
class Loss(Function):
    """Base class of loss functions."""

    def __init__(self):
        super().__init__()
    
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
    def forward(self, y, t):
        """Compute the loss between the y and the t."""
        raise NotImplementedError
    
    @abstractmethod
    def backward(self):
        """Compute the gradient of the previous loss."""
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
        super().__init__()
        self.p = p
        
    def forward(self, y, t):
        """Compute the p-th power of the p-norm of the residuals."""
        self._res = res = y - t  # residuals
        loss = np.sum((np.abs(res) if self.p % 2 else res) ** self.p)
        return loss
    
    def backward(self):
        if self.p == 1:
            return sign(self._res)
        elif self.p == 2:
            return self._res
        elif self.p % 2:
            return sign(self._res) * self._res ** (self.p - 1)
        else:
            return self._res ** (self.p - 1)


class CrossEntropy(Loss):
    """Cross entropy loss, usually used in classification."""

    def __init__(self):
        super().__init__()
    
    def forward(self, y, t):
        self._y, self._t = y, t
        return - t @ np.log(y)
    
    def backward(self):
        return - self._t / self._y
    
    
class SoftMaxCE(Loss):
    """Cross entropy with softmax transformation."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, y, t):
        """The y and the t should be probability distributions."""
        self._y, self._t = y, t
        y = y - np.max(y, axis=-1, keepdims=True)
        exp_sum = np.sum(np.exp(y), axis=-1)
        dot_prod = np.sum(y * t, axis=-1)
        return np.sum(np.log(exp_sum) - dot_prod)
    
    def backward(self, y, t):
        return self._y - self._t


class Activation(Function):
    """Activation function to add nonlinearity."""
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def backward(self, error):
        "Compute the gradient of the error w.r.t the previous output."
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
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return np.tanh(x)

    def backward(self, err):
        return err * (1 - self._ret**2)


class Logistic(Activation):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, err):
        return err * self._ret * (1 - self._ret)


class ReLU(Activation):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, err):
        return err * (self._ret > 0)


class SoftMax(Activation):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return ex / np.sum(ex, axis=-1, keepdims=True)
    
    def backward(self, err):
        # TODO: not correct?
        dp = np.sum(err * self._ret, axis=-1, keepdims=True)
        return (err - dp) * self._ret
    
    
if __name__ == '__main__':
    relu = ReLU()
    relu(np.random.rand(10, 3) - 0.5)
    print(relu.backward(np.random.rand(10, 3)))
    
    sm = SoftMax()
    sm(np.random.rand(10, 3))
    err = np.random.rand(10, 3)
    for y, e, be in zip(sm._ret, err, sm.backward(err)):
        d1 = np.diag(y) - np.outer(y, y)
        assert (d1 @ e == be).all()
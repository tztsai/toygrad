from function import Function
from utils import *


class Loss(Function):
    """Base class of loss functions."""

    def __new__(cls, obj):
        if type(obj) is str:
            s = obj.lower()
            if s[0] == 'l' and s[1:].isdigit():
                return L(int(s[1:]))
            elif s in ['crossentropy', 'cross_entropy', 'ce']:
                return CrossEntropy()
            elif s in ['softmax_crossentropy', 'softmax_ce',
                       'softmax_cross_entropy', 'smce']:
                return SoftMaxCrossEntropy()
            else:
                raise ValueError(f"unknown loss function: {obj}")
        elif isinstance(obj, cls):  # already an instance of Loss
            return obj
        else:
            raise ValueError(f"unknown loss function: {obj}")

    @abstractmethod
    def forward(self, y, t):
        """Compute the loss between the output `y` and the target `t`."""
        raise NotImplementedError

    @abstractmethod
    def backward(self):
        """Compute the gradient of the previous loss w.r.t the output."""
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
        self.res = res = y - t  # residuals
        loss = np.sum((np.abs(res) if self.p % 2 else res) ** self.p)
        return loss

    def backward(self):
        if self.p == 1:
            return sign(self.res)
        elif self.p == 2:
            return self.res
        elif self.p % 2:
            return sign(self.res) * self.res ** (self.p - 1)
        else:
            return self.res ** (self.p - 1)


class CrossEntropy(Loss):
    """Cross entropy loss, usually used in classification."""
    
    def forward(self, y, t):
        self.y, self.t = y, t
        return - t @ np.log(y)

    def backward(self):
        return - self.t / self.y


class SoftMaxCrossEntropy(Loss):
    """Cross entropy with softmax transformation."""

    def forward(self, y, t):
        """The y and the t should be probability distributions."""
        self.y, self.t = y, t
        y = y - np.max(y, axis=-1, keepdims=True)
        exp_sum = np.sum(np.exp(y), axis=-1)
        dot_prod = np.sum(y * t, axis=-1)
        return np.sum(np.log(exp_sum) - dot_prod)

    def backward(self):
        return self.y - self.t

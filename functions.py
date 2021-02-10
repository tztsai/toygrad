import numpy as np


def Lp_loss(output, target, metric='l2', average=True):
    target = np.reshape(target, [len(target), -1])

    if metric == 'l2':
        loss = np.sum((output - target) ** 2)
    elif metric == 'l1':
        loss = np.sum(np.abs(output - target))
    else:
        raise ValueError('unknown loss metric')

    if average: loss /= len(output)
    return loss


def get_activation(name: str):
    name = name.lower()
    if name == 'tanh':
        return Tanh()
    elif name == 'logistic':
        return Logistic()
    elif name == 'relu':
        return ReLU()
    elif name == 'linear':
        return None
    else:
        raise ValueError('unknown activation function "%s"' % name)


def is_activation(obj):
    return isinstance(obj, Activation) or obj in [False, None]
    
    
class Activation:
    def __call__(self, x):
        raise NotImplementedError

    def deriv(self, y, activated=True):
        """
        Return the derivative value of the activation.
        
        Args:
            y: the input value at which the derivative is calculated
            activated: whether the input has been activated
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
    def __call__(self, x):
        return np.maximum(x, 0)

    def deriv(self, y, activated=True):
        return (y > 0).astype(float)


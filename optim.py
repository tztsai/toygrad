from utils.dev import *


@parse_name
def get_optimizer(name: str):
    if name == 'sgd':
        return SGD
    elif name == 'adam':
        raise NotImplementedError
    else:
        raise ValueError(f"unknown optimizer: {name}")
    
OptimGetter = makemeta(get_optimizer)


@DefaultNone
class Optimizer(ABC, metaclass=OptimGetter):
    """Base class of an optimizer."""
    
    def update(self, parameters):
        """Update weights in the whole neural network."""
        for param in parameters:
            if not param.grad_zero:
                param += self.delta(param)
                param.zero_grad()

    @abstractmethod
    def delta(self, parameter):
        """The change rate of a parameter given by the optimizer."""
        raise NotImplementedError


class SGD(Optimizer):
    learning_rate = 1e-3
    momentum = 0.8
    
    def __init__(self, lr=None, momentum=None):
        self.learning_rate = lr
        self.momentum = momentum
        self.old_delta = {}
        
    def delta(self, parameter):
        if parameter in self.old_delta:
            delta = (self.momentum * self.old_delta[parameter] - 
                     (1 - self.momentum) * parameter.grad)
        else:
            delta = -parameter.grad
        self.old_delta[parameter] = delta
        return self.learning_rate * delta

    def __repr__(self):
        clsname = type(self).__name__
        return f'{clsname}(lr={self.learning_rate:.2e}), momentum={self.momentum:.2f})'


if __name__ == '__main__':
    print(Optimizer('SGD'))
    sgd = SGD(0.1, 0.2)
    print(sgd)
    sgd.momentum = None
    print(sgd)
    # print(Optimizer())
    print(Optimizer('sgd', 2))

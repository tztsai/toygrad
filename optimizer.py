from utils import abstractmethod, none_for_default, name2obj


def get_optimizer(name):
    if name == 'sgd':
        return object.__new__(SGD)
    elif name == 'adam':
        raise NotImplementedError
    else:
        raise ValueError(f"unknown optimizer: {name}")


@none_for_default
class Optimizer(metaclass=name2obj(get_optimizer)):
    """Base class of an optimizer."""
    learning_rate = 1e-2
    
    def update(self, parameters):
        """Update weights in the whole neural network."""
        for param in parameters:
            if param.need_update:
                param += self.learning_rate * self.grad(param)
                param.zero_grad()

    @abstractmethod
    def grad(self, parameter):
        """The change rate of a parameter given by the optimizer."""
        raise NotImplementedError
    
    def __repr__(self):
        return type(self).__name__ + f'(lr={self.learning_rate:.2e})'
    

class SGD(Optimizer):
    momentum = 0.8
    
    def __init__(self, lr=None, momentum=None):
        self.learning_rate = lr
        self.momentum = momentum
        
    def grad(self, parameter):
        if not hasattr(parameter, 'prev_grad'):
            grad = -parameter.grad
        else:
            grad = (self.momentum * parameter.prev_grad - 
                    (1 - self.momentum) * parameter.grad)
        parameter.prev_grad = grad  # store the gradient for the next update
        return grad

    def __repr__(self):
        return super().__repr__()[:-1] + f', momentum={self.momentum:.2f})'


print(Optimizer('sgd'))
print(SGD(0.1))
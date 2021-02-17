from utils import baseclass, abstractmethod, Default


class Optimizer(baseclass):
    """Base class of an optimizer."""
    learning_rate = Default(1e-2)
    
    def __init__(self, lr=None):
        self.learning_rate = lr
    
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
    
    @classmethod
    def get(cls, opt, lr=None):
        if type(opt) is str:
            opt = opt.lower()
            if opt == 'sgd':
                return SGD(lr)
        elif isinstance(opt, cls):
            return opt
        else:
            raise ValueError(f"unknown optimizer: {opt}")


class SGD(Optimizer):
    momentum = Default(0.8)
    
    def __init__(self, lr=None, momentum=None):
        super().__init__(lr)
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

from utils import baseclass, abstractmethod


class Optimizer(baseclass):
    """Base class of an optimizer."""
    learning_rate = 1e-2
    
    def __init__(self, lr=learning_rate):
        self.lr = lr
    
    def update(self, parameters):
        """Update weights in the whole neural network."""
        for param in parameters:
            if param.need_update:
                param += self.lr * self.grad(param)
                param.zero_grad()

    @abstractmethod
    def grad(self, parameter):
        """The change rate of a parameter given by the optimizer."""
        raise NotImplementedError
    
    def __repr__(self):
        return type(self).__name__ + '(lr=%.2e)' % self.lr


class SGD(Optimizer):
    learning_rate = 1e-3
    momentum = 0.8
    
    def __init__(self, lr=learning_rate, momentum=momentum):
        super().__init__(lr)
        self.mo = momentum

    def grad(self, parameter):
        if not hasattr(parameter, 'prev_grad'):
            grad = -parameter.grad
        else:
            grad = self.mo * parameter.prev_grad - (1 - self.mo) * parameter.grad
        parameter.prev_grad = grad  # store the gradient for the next update
        return grad

    def __repr__(self):
        return super().__repr__()[:-1] + ', momentum=%.2f)' % self.mo
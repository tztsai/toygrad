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
                delta = self.delta(param)
                param += self.lr * delta
                param.delta = delta
                param.zero_grad()

    @abstractmethod
    def delta(self, parameter):
        """The change of a parameter before scaling by learning rate."""
        raise NotImplementedError
    
    def __repr__(self):
        return type(self).__name__ + '(lr=%.2e)' % self.lr


class SGD(Optimizer):
    learning_rate = 1e-3
    momentum = 0.8
    
    def __init__(self, lr=learning_rate, momentum=momentum):
        super().__init__(lr)
        self.mo = momentum

    def delta(self, parameter):
        return self.mo * parameter.delta - (1 - self.mo) * parameter.grad

    def __repr__(self):
        return super().__repr__()[:-1] + ', momentum=%.2f)' % self.mo
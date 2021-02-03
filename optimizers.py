class Optimizer:
    """Base class of an optimizer"""
    learning_rate = 1e-3
    
    def __init__(self, learning_rate=learning_rate):
        self.lr = learning_rate
    
    def update(self, parameters):
        """
        Update weights in the whole neural network.
        """
        for param in parameters:
            delta = self.delta(param)
            param += self.lr * delta
            param.delta = delta
            param.zero_grad()

    def delta(self, parameter):
        """
        Return the change of a parameter before scaling by learning rate.
        """
        raise NotImplementedError

    
class SGD(Optimizer):
    learning_rate = 1e-3
    momentum = 0.8
    
    def __init__(self, learning_rate=learning_rate, momentum=momentum):
        super().__init__(learning_rate)
        self.mo = momentum

    def delta(self, parameter):
        return self.mo * parameter.delta - (1 - self.mo) * parameter.grad

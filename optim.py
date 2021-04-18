from core import Param, AbstractFunction, np
from utils.dev import DefaultNone, abstractmethod, info, defaultdict


class Optimizer(AbstractFunction):
    """Base class of an optimizer."""

    lr = 1e-3    # learning rate
    decay = None # weight decay
    lamb = 2e-3  # coefficient of weight decay (lambda)

    def __init__(self, lr=lr, decay=decay, lamb=lamb):
        self.lr = lr
        self.decay = decay
        self.lamb = lamb

    def apply(self, parameters):
        for param in parameters:
            assert isinstance(param, Param)
            if not param.grad_clean:
                if param.trainable:
                    param += self.delta(param)
                    if self.decay:
                        param -= self.lr * self.lamb * self.decay_term(param)
                param.zero_grad()

    @abstractmethod
    def delta(self, parameter):
        """The amount of change of a parameter given by the optimizer."""
        
    def decay_term(self, param):
        if self.decay is None:
            return 0
        elif callable(self.decay):
            return self.decay(param)
        elif type(self.decay) is str:
            if self.decay.lower() in ['l1', 'lasso']:
                return np.sign(param)
            elif self.decay.lower() in ['l2', 'ridge']:
                return param
            else:
                raise ValueError('unknown regularization')
        else:
            raise TypeError('invalid regularization type')

class SGD(Optimizer):
    lr = 1e-3
    m = 0.8
    
    def __init__(self, lr=lr, mom=m, **kwds):
        super().__init__(lr, **kwds)
        self.m = mom
        self.old_delta = {}
        
    def delta(self, p):
        if p in self.old_delta:
            delta = self.m * self.old_delta[p] - (1-self.m) * p.grad
        else:
            delta = -p.grad
        self.old_delta[p] = delta
        return self.lr * delta

class Adam(Optimizer):
    lr = 1e-3
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
        
    def __init__(self, lr=lr, b1=b1, b2=b2, eps=eps, **kwds):
        super().__init__(lr, **kwds)
        self.b1, self.b2, self.eps = b1, b2, eps
        self.t, self.m, self.v = 0, defaultdict(float), defaultdict(float)

    def apply(self, params):
        self.t += 1
        super().apply(params)

    def delta(self, p):
        self.m[p] = self.b1 * self.m[p] + (1. - self.b1) * p.grad
        self.v[p] = self.b2 * self.v[p] + (1. - self.b2) * p.grad**2
        m = self.m[p] / (1. - self.b1 ** self.t)
        v = self.v[p] / (1. - self.b2 ** self.t)
        return -self.lr * m / (np.sqrt(v) + self.eps)

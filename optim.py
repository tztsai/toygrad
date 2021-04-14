from core import Param, AbstractFunction, np
from utils.dev import DefaultNone, abstractmethod, info, defaultdict


class Optimizer(AbstractFunction):
    """Base class of an optimizer."""

    lr = 1e-3    # learning rate
    reg = None   # regularization
    lamb = 1e-3  # regularization coefficient (lambda)

    def __init__(self, lr=lr, reg=reg, lamb=lamb):
        self.lr = lr
        self.reg = reg
        self.lamb = lamb

    def apply(self, parameters):
        for param in parameters:
            if not isinstance(param, Param):
                info('Warning: %s is not a Param, skip optimization', param)
            elif not param.grad_zero:
                param += self.delta(param)
                if self.reg: param += self.reg_term(param)
                param.zero_grad()

    @abstractmethod
    def delta(self, parameter):
        """The amount of change of a parameter given by the optimizer."""
        
    def reg_term(self, param):
        if self.reg is None:
            return 0
        elif callable(self.reg):
            return self.lamb * self.reg(param)
        elif type(self.reg) is str:
            if self.reg.lower() in ['l1', 'lasso']:
                return self.lamb * -np.sign(param)
            elif self.reg.lower() in ['l2', 'ridge']:
                return self.lamb * -param
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
    lr = 5e-3
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
        
    def __init__(self, lr=lr, b1=b1, b2=b2, eps=eps, **kwds):
        super().__init__(lr, **kwds)
        self.b1, self.b2, self.eps = b1, b2, eps
        self.t, self.m, self.v = 0, defaultdict(float), defaultdict(float)

    def delta(self, p):
        self.t += 1
        a = self.lr * ((1. - self.b2**self.t)**0.5) / (1. - self.b1**self.t)
        self.m[p] = self.b1 * self.m[p] + (1. - self.b1) * p.grad
        self.v[p] = self.b2 * self.v[p] + (1. - self.b2) * p.grad**2
        return -a * self.m[p] / (np.sqrt(self.v[p]) + self.eps)

from .core import Function, Param, np
from .utils.dev import abstractmethod, info, defaultdict


class Optimizer(Function):
    """ Base class of an optimizer. """

    lr = 1e-3    # learning rate
    reg = None   # regularization
    lamb = 2e-3  # coefficient of regularization (lambda)
    grad_lim = None  # limit of the max magnitude of numbers in a gradient

    def __new__(cls, lr=lr, *, reg=None, lamb=None, **kwds):
        kwds.update((k, v) for k, v in locals().items()
                    if k not in ['cls', 'kwds'] and v is not None)
        opt = cls.new((), kwds)
        opt.__dict__.update(kwds)
        return opt
        
    def __call__(self, parameters):
        with Param.not_training():
            for par in parameters:
                assert isinstance(par, Param) and not par.constant
                if self.grad_lim:
                    par.shrink_grad(self.grad_lim)
                par += self.delta(par)
                if self.reg:
                    par += self.lr * self.lamb * self.reg_term(par)
                par.zero_grad()

    @abstractmethod
    def delta(self, parameter):
        """The amount of change of a parameter given by the optimizer."""

    def reg_term(self, param):
        if self.reg is None:
            return 0
        elif callable(self.reg):
            return self.reg(param)
        elif type(self.reg) is str:
            if self.reg.lower() in ['l1', 'lasso']:
                return -np.sign(param)
            elif self.reg.lower() in ['l2', 'ridge']:
                return -param
            else:
                raise ValueError('unknown regularization')
        else:
            raise TypeError('invalid regularization type')

class SGD(Optimizer):
    lr = 1e-3
    mom = 0.8
    
    def __init__(self, lr=lr, *, mom=mom, **kwds):
        self.old_delta = {}
        
    def delta(self, p):
        if p in self.old_delta:
            delta = self.mom * self.old_delta[p] - (1-self.mom) * p.grad
        else:
            delta = -p.grad
        self.old_delta[p] = delta
        return self.lr * delta

class Adam(Optimizer):
    lr = 1e-3
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8

    def __init__(self, lr=lr, *, b1=None, b2=None, eps=None, **kwds):
        self.t, self.m, self.v = 0, defaultdict(float), defaultdict(float)

    def __call__(self, params):
        self.t += 1
        super().__call__(params)

    def delta(self, p):
        self.m[p] = self.b1 * self.m[p] + (1. - self.b1) * p.grad
        self.v[p] = self.b2 * self.v[p] + (1. - self.b2) * p.grad**2
        m = self.m[p] / (1. - self.b1 ** self.t)
        v = self.v[p] / (1. - self.b2 ** self.t)
        return -self.lr * m / (np.sqrt(v) + self.eps)

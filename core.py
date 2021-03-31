import numpy as np
from utils.dev import *


class Parameter(np.ndarray):
    """A trainable parameter."""

    training = True
    rng = np.random.default_rng()
    grad_lim = 1e8  # magnitude limit of each element of the gradient

    def __new__(cls, value=None, *, size=None, mean=0, scale=None,
                dtype=np.float, learnable=True):
        """
        If `value` is given, then it will be converted to a Parameter.
        If `dtype` is the same as that of the given `value`, then a view of
        `value` will be returned, so its data will not be copied.
        However, if `size` is additionally specified, then a new Parameter
        of this size will be created filled with the given `value`.
        
        If `value` is not given, a random Parameter following normal
        distribution will be generated. Additionally, `mean` and `scale`
        of the distribution can be specified.
        
        >> Parameter([[1,2,3],[4,5,6]])
        >> Parameter(size=[4, 4], dtype=np.float32, scale=1)
        >> Parameter(0, size=[5, 5])
        >> w = Parameter(size=[5, 3])
        >> w is Parameter(w)
        """
        if value is None:  # random initialization
            if size is None:
                size = 1
            if scale is None:
                length = size[0] if hasattr(size, '__len__') else size
                scale = 1 / np.sqrt(length)  # Xavier initialization
            value = cls.rng.normal(size=size, loc=mean, scale=scale)
        else:  # convert the value to an array
            if size is not None:  # fill an array of the given size
                value = np.full(size, value, dtype=dtype)
                
        param = np.asarray(value, dtype=dtype).view(cls)
        param.learnable = learnable
        param._ctx = None
        param._grad = 0 if learnable else None
        return param

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, grad):
        if not self.learnable: return
        elif not np.shape(grad):  # a scalar
            grad = np.full(self.shape, grad)
        elif np.shape(grad) != self.shape:
            raise ValueError('gradient shape mismatch')
        self._grad = np.clip(grad, -self.grad_lim, self.grad_lim)

    def zero_grad(self):
        self._grad = 0
        
    def fill(self, value):
        self[:] = np.full_like(self, value)
        
    @property
    def grad_zero(self):
        return id(self.grad) == id(0)
    
    def backward(self):
        def dfs(y, visited={None}):
            ctx = y._ctx
            if ctx in visited: return
            visited.add(ctx)
            
            derivs = [ctx.deriv] if len(ctx.parents) == 1 else ctx.deriv
            assert len(derivs) == len(ctx.parents)
            
            for x, deriv in zip(ctx.parents, derivs):
                if not isinstance(x, Parameter): continue
                # ∂e/∂x = ∂y/∂x · ∂e/∂y  (e is the source of backward pass)
                if ctx.dim == 0:  # TODO: make it work for minibatches
                    x.grad = deriv * y.grad
                elif ctx.dim == 1:
                    x.grad = deriv @ y.grad
                else:
                    raise NotImplementedError
                if isinstance(x, Parameter):
                    dfs(x, visited)
                    
        self.grad = 1  # the gradient of the source param wrt itself is constant 1
        dfs(self)

    def __hash__(self):
        return id(self)
        

class Operation(type):
    """A metaclass of Parameter operations."""
        
    class AbstractOp:
        """
        The baseclass of Parameter operations.
        An instantiation of it creates a context in the computation graph.
        """
        dim = 0  # the minimum dimension of the operation
        # dim = 0: +, *, exp, ReLU, tanh, ...
        # dim = 1: @, ...
        # dim = 2: Conv2D, ...
        
        def __new__(cls, *args, **kwds):
            ctx = object.__new__(cls)
            ctx.parents = args
            ctx.deriv = None
            with ProfileOp(cls.__name__, args):
                return ctx(*args, **kwds)

        @abstractmethod
        def apply(self, *args, **kwds):
            """Computes the output and stores its derivative matrix in `self.deriv`."""
            raise NotImplementedError

        def __call__(self, *args, **kwds):
            """Wraps the apply method to process arguments and the return value."""
            learnable = any(isinstance(p, Parameter) and p.learnable for p in args)

            try:  # unify types
                p0 = next(arg for arg in args if isinstance(arg, Parameter))
                dtype = p0.dtype
            except StopIteration:  # no parameters in the arguments
                return self.apply(*args, **kwds)
            
            # make sure all inputs are not Parameter objects
            args = [np.asarray(arg, dtype=dtype) if isinstance(arg, Parameter)
                    else arg for arg in args]
            assert not any(isinstance(val, Parameter) for val in kwds.values())
            
            result = Parameter(self.apply(*args, **kwds), dtype=dtype, learnable=learnable)
            if learnable: result._ctx = self
            return result

        def __repr__(self):
            return f"{type(self).__name__}({', '.join(map(repr, self.parents))})"
    
    def __new__(meta, name, bases, dict):
        op = type.__new__(meta, name, (meta.AbstractOp,), dict)
        
        # convert the operation to a function (otherwise it cannot be an instance method)
        def f(*args, **kwds): return op(*args, **kwds)
        
        # register the operation in the Parameter class
        name = name.lower()
        setattr(Parameter, name, f)
        if name in ['add', 'sub', 'mul', 'pow', 'matmul']:
            setattr(Parameter, f"__{name}__", f)
            setattr(Parameter, f"__r{name}__", lambda self, x: f(x, self))
            setattr(Parameter, f"__i{name}__", lambda self, x: self.fill(f(self, x)))
            
        return op
    
    def __repr__(self):
        return self.__name__

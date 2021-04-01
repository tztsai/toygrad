import numpy as np
import scipy as sp
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
        else:
            if size is not None:  # fill an array of the given size
                value = np.full(size, value, dtype=dtype)

        param = np.asarray(value, dtype=dtype).view(cls)
        param.learnable = learnable
        param._ctx = None
        param._grad = 0 if learnable else None
        return param

    @property
    def grad(self): return self._grad

    @grad.setter
    def grad(self, grad):
        if not self.learnable: return
        elif not np.shape(grad):  # a scalar
            grad = np.full(self.shape, grad)
        elif np.shape(grad) != self.shape:
            raise ValueError('gradient shape mismatch')
        self._grad = np.clip(grad, -self.grad_lim, self.grad_lim)
        
    @property
    def grad_zero(self): return id(self.grad) == id(0)

    def zero_grad(self): self._grad = 0
        
    def fill(self, value): self[:] = np.full_like(self, value)

    def backward(self):
        def dfs(param, visited={None}):
            ctx = param._ctx
            if ctx in visited: return
            # print(ctx)
            visited.add(ctx)
            grads = ctx.backward(param)
            for par, grad in zip(ctx.parents, grads):
                if isinstance(par, Parameter) and par.learnable:
                    par.grad += grad
                    dfs(par, visited)
        assert not self.shape, 'backprop must start from a scalar'
        self.grad = 1  # the gradient of the source param wrt itself is constant 1
        dfs(self)

    def __hash__(self): return id(self)
        

class Operation(type):
    """A metaclass of Parameter operations."""
        
    class AbstractOp:
        """
        The baseclass of Parameter operations.
        An instantiation of it creates a context in the computation graph.
        """
        ndim_in, ndim_out = 0, 0  # num of dims of input and output on which op is applied
        omitted_axes = ()  # the input axes omitted in deriv, some may be bound to output axes
        
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
        
        def _passgrad(self, y, ix):
            """Computes the gradient of the parent no.`ix` given the child `y`.
                1. expand grad: insert new axes into the gradient of the output y
                2. swap: swap the "constrained" axes of y with the correspinding new axes
                3. squeeze: remove the swapped new axes
                4. multiply: multiply the gradient of y with the partial derivatives of op
                5. sum: sum up and eliminate the tail axes that corresponds to the axes of y
                6. expand deriv: insert omitted exes into the deriv array dy/dx
                7. debroadcast: sum up (but not remove) the broadcasted axes of the gradient of x
            """
            def splitshape(sh, k): return sh[:k], sh[k:]
            def axes(a, b): return tuple(range(a, b))
            def getattrs(*ss): return \
                [(a:=getattr(self, s), a if len(self.parents) == 1 else a[ix])[1] for s in ss]
            x = self.parents[ix]
            xdim,dydx,bnd_axes,omt_axes = getattrs('ndim_in','deriv','bound_axes','omitted_axes')
            ydim = self.ndim_out            
            xsh_ext, xsh = splitshape(x.shape, -xdim)
            ysh_ext, ysh = splitshape(y.shape, -ydim)
            
            bc_axes = []  # broadcasted axes of x
            for i in range(len(xsh_ext)):
                j, k = x.shape[-xdim-i], y.shape[-ydim-i]
                if j == 1 and k > 1: bc_axes.append(-xdim-i)
                else: assert j == k

            ofs1, ofs2 = -xdim-ydim, -ydim
            dy = np.expand_dims(y.grad, axes(ofs1, ofs2))
            dydx_axes_to_expand = list(omt_axes)
            for a1, a2 in enumerate(omt_axes):
                if type(a2) is int:  # bound to output axis a2
                    dy = np.swapaxes(dy, a1+ofs1, a2+ofs2)
                elif not a2:  # axis not omitted
                    dydx_axes_to_expand.remove(a1)
            dydx = np.expand_dims(dydx, axis=tuple(dydx_axes_to_expand))

            # basically, ∂e/∂x = Σ_y (∂e/∂y * ∂y/∂x)
            grad = np.sum(dy * dydx, axis=axes(ofs2, 0))
            # debroadcast - sum over the broadcasted axes
            grad = np.sum(grad, axis=tuple(bc_axes), keepdims=True)
            grad = np.squeeze(grad, axis=axes(-grad.ndim, -x.ndim))
            assert grad.shape == x.shape
            return grad
        
        def backward(self, child):
            """Given the child node, computes the gradients of the parents.
               Override this to define your own backprop method.
               You may want to save something for the backprop when you apply the operation -
               just add any attribute to `self` (its name better bigins with '_')."""
            for i in range(len(self.parents)): yield self._passgrad(child, i)

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
            
            output = Parameter(self.apply(*args, **kwds),
                               dtype=dtype, learnable=learnable)
            if learnable: output._ctx = self
            return output

        def __repr__(self):
            return f"{type(self).__name__}({', '.join(map(repr, self.parents))})"
    
    def __new__(meta, name, bases, dict):
        op = type.__new__(meta, name, (meta.AbstractOp,), dict)
        # convert the class to a function (otherwise it cannot be a method)
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

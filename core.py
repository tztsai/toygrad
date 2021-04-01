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

    def __hash__(self):
        return id(self)
        

class Operation(type):
    """A metaclass of Parameter operations."""
        
    class AbstractOp:
        """
        The baseclass of Parameter operations.
        An instantiation of it creates a context in the computation graph.
        """
        in_dim, out_dim = 0, 0
        broadcastable = False
        # dimensionalities of input and output on which the operation is applied
        
        def __new__(cls, *args, **kwds):
            ctx = object.__new__(cls)
            ctx.parents = args
            ctx.deriv = None
            ctx.bound_axes = ()  # XXX: Make this Clear!
            with ProfileOp(cls.__name__, args):
                return ctx(*args, **kwds)

        @abstractmethod
        def apply(self, *args, **kwds):
            """Computes the output and stores its derivative matrix in `self.deriv`."""
            raise NotImplementedError
        
        def _passgrad(self, y, x, in_dim, deriv):
            """6 steps to compute the gradient of the input x:
            1. expand: insert new axes into the gradient of the output y
            2. swap: swap the "constrained" axes of y with the correspinding new axes
            3. squeeze: remove the swapped new axes
            4. multiply: multiply the gradient of y with the partial derivatives of op
            5. sum: sum up and eliminate the tail axes that corresponds to the axes of y
            6. debroadcast: sum up (but not remove) the broadcasted axes of the gradient of x
            """
            def splitshape(sh, k): return sh[:k], sh[k:]
            def axes(a, b): return tuple(range(a, b))
            
            x_sh_ext, x_sh = splitshape(x.shape, -in_dim)
            y_sh_ext, y_sh = splitshape(y.shape, -out_dim)
            
            sum_axes = []
            if self.broadcastable:  # find the broadcasted axes
                assert len(x_sh_ext) == len(y_sh_ext)
                for i, (j, k) in enumerate(zip(x_sh_ext, y_sh_ext)):
                    if j == 1 and k > 1: sum_axes.append(j)
                    else: assert j == k
            else:
                assert x_sh_ext == y_sh_ext

            # d_sh_ext, d_sh = splitshape(deriv.shape, len(x_sh_ext))
            # assert d_sh_ext == x_sh_ext

            ofs1, ofs2 = -in_dim-out_dim, -out_dim
            y_grad_ex = np.expand_dims(y.grad, axes(ofs1, ofs2))
            swapped_axes = []
            for a2, a1 in enumerate(self.bound_axes):
                if a1 >= 0:  # output axis a2 is bound to input axis a1
                    swapped_axes.append(a2+ofs2)
                    y_grad_ex = np.swapaxes(y_grad_ex, a1+ofs1, a2+ofs2)
            y_grad_ex = np.squeeze(y_grad_ex, axis=tuple(swapped_axes))
            grad = np.sum(y_grad_ex * deriv, axis=axes(len(swapped_axes)+ofs2, 0))
            # ed1, ed2 = self.bound_dim-len(d_sh), self.bound_dim-out_dim
            # assert d_sh[:ed2] == x_sh and d_sh[ed2:] == y_sh[ed2:]
            # y_grad_ex = np.expand_dims(y.grad, axes(ed1, ed2))
            # grad = np.sum(y_grad_ex * deriv, axis=axes(ed2, 0))
            
            # if ndd == ndx + ndy:
            #     assert d_sh == x_sh + y_sh
            #     y_grad = np.expand_dims(y.grad, tuple(range(-ndd, -ndy)))
            #     grad = np.sum(y_grad * deriv, axis=tuple(range(-ndy, 0)))

            grad = np.sum(grad, axis=sum_axes, keepdims=True)  # debroadcast
            assert grad.shape == x.shape
            return grad
        
        def backward(self, child):
            """Given the child node, computes the gradients of the parents.
               Override this to define your own backprop method.
               You may want to save something for the backprop when you apply the operation -
               just add any attribute to `self` (its name better bigins with '_').""" 
            if len(self.parents) == 1:
                in_dims = [self.in_dim]
                derivs = [self.deriv]
            for par, ind, deriv in zip(self.parents, in_dims, derivs):
                yield self._passgrad(par, child, ind, deriv)

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
            if self.deriv is None:
                return f"{type(self).__name__}({', '.join(map(repr, self.parents))})"
            else:
                return f"{type(self).__name__}'({', '.join(map(repr, self.parents))})={self.deriv}"
    
    def __new__(meta, name, bases, dict):
        op = type.__new__(meta, name, (meta.AbstractOp,), dict)
        assert len(op.bound_axes) == op.out_dim and \
            all(a == -1 or a in range(op.in_dim) for a in op.bound_axes)
        
        # convert the operation to a function (otherwise it cannot be an instance method)
        def f(*args, **kwds): return op(*args, **kwds)
        
        # register the operation in the Parameter class
        name = name.lower()
        setattr(Parameter, name, f)
        if name in ['add', 'sub', 'mul', 'pow', 'matmul']:
            setattr(Parameter, f"__{name}__", f)
            setattr(Parameter, f"__r{name}__", lambda self, x: f(x, self))
            setattr(Parameter, f"__i{name}__", lambda self, x: self.fill(f(self, x)))
            op.broadcastable = True
        return op
    
    def __repr__(self):
        return self.__name__

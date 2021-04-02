import numpy as np
from utils.dev import ABCMeta, ABC, abstractmethod, ProfileOp


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
        elif np.ndim(grad) == 0:  # a scalar
            grad = np.full(self.shape, grad)
        elif np.shape(grad) != self.shape:
            raise ValueError('gradient shape mismatch')
        self._grad = np.clip(grad, -self.grad_lim, self.grad_lim)
        
    @property
    def grad_zero(self): return id(self.grad) == id(0)

    def zero_grad(self): self._grad = 0
        
    def fill(self, value):
        value = np.full_like(self, value)
        if self.shape: self[:] = value
        else: return Parameter(self, value, learnable=self.learnable)

    def backward(self):
        def dfs(param, visited={None}):
            ctx = param._ctx
            if ctx in visited: return
            # print(ctx)
            visited.add(ctx)
            grads = ctx.backward(param)
            for par, grad in zip(ctx.inputs, grads):
                if isinstance(par, Parameter) and par.learnable:
                    par.grad += grad
                    dfs(par, visited)
        assert not self.shape, 'backprop must start from a scalar'
        self.grad = 1  # the gradient of the source param wrt itself is constant 1
        dfs(self)

    def __hash__(self): return id(self)
        

class Operation(ABCMeta):
    """A metaclass of Parameter operations."""
        
    class AbstractOp(ABC):
        """
        The baseclass of Parameter operations.
        An instantiation of it creates a context in the computation graph.
        """
        ndim_in, ndim_out = 0, 0  # num of dims of input and output on which op is applied
        bound_axes = ()  # pairs of input-output axes that must be identical in deriv
        omitted_axes = ()  # the input axes omitted in deriv
        
        def __new__(cls, *args, **kwds):
            ctx = object.__new__(cls)
            ctx.inputs = args
            ctx.deriv = None
            with ProfileOp(cls.__name__, args):
                return ctx(*args, **kwds)

        @abstractmethod
        def apply(self, *args, **kwds):
            """Computes the output and stores its derivative matrix in `self.deriv`."""
            raise NotImplementedError
        
        def _passgrad(self, y, ix):
            """Computes the gradient of the input no.`ix` given the output `y`.
            1. expand grad: insert new axes into the output gradient de/dy
            2. swap: swap the "constrained" axes of de/dy with the corresponding new axes
            3. squeeze: remove the swapped new axes of de/dy
            3. expand deriv: insert omitted exes into the partial derivatives dy/dx
            4. multiply: multiply de/dy with dy/dx
            5. sum: sum up the product along axes corresponding to the output y
            6. debroadcast: sum up along the broadcasted axes of the input gradient de/dx
            """
            def getattrs(*ss): return \
                [(a:=getattr(self, s), a if len(self.inputs) == 1 else a[ix])[1] for s in ss]
            x = self.inputs[ix]
            xdim,dyx,bnd_axes,omt_axes = getattrs('ndim_in','deriv','bound_axes','omitted_axes')
            ydim = self.ndim_out            
            n_bnd = len(bnd_axes)

            off1, off2 = -xdim-ydim, -ydim
            dy = np.expand_dims(y.grad, tuple(range(off1, off2)))
            for a1, a2 in bnd_axes: dy = np.swapaxes(dy, a1+off1, a2+off2)
            dy = np.squeeze(dy, axis=tuple(a2+off2 for _, a2 in bnd_axes))
            dyx = np.expand_dims(dyx, axis=tuple(a1+n_bnd+off1 for a1 in omt_axes))
            # basically, ∂e/∂x = Σ_y (∂e/∂y * ∂y/∂x)
            grad = np.sum(dy * dyx, axis=tuple(range(n_bnd + off2, 0)))

            def backsplit(sh, k): return (sh[:-k], sh[-k:]) if k else (sh, ())
            xsh_ext, xsh = backsplit(x.shape, xdim)
            gsh_ext, gsh = backsplit(grad.shape, xdim)
            assert gsh == xsh
            bc_axes = []  # broadcasted axes of x
            for i in range(len(gsh_ext)):
                k = gsh_ext[-i-1]
                try: j = xsh_ext[-i-1]
                except IndexError: j = 1
                if j == 1 and k > 1: bc_axes.append(len(gsh_ext)-i-1)
                else: assert j == k
            # debroadcast - sum over the broadcasted axes
            grad = np.sum(grad, axis=tuple(bc_axes))
            
            return grad.reshape(x.shape)
        
        def backward(self, output):
            """Given the output node, computes the gradients of the inputs.
            Override this to define your own backprop method.
            You may want to save something for the backprop when you apply the operation -
            just add any attribute to `self` (its name better bigins with '_').
            Note that the return value should be iterable - you can yield a single grad
            """
            return [self._passgrad(output, i) for i, _ in enumerate(self.inputs)]

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
            return f"{type(self).__name__}({', '.join(map(repr, self.inputs))})"
    
    def __new__(meta, name, bases, dict):
        op = super().__new__(meta, name, (meta.AbstractOp,), dict)
        # convert the class to a function (otherwise it cannot be a method)
        def f(*args, **kwds): return op(*args, **kwds)
        # register the operation in the Parameter class
        name = name.lower()
        setattr(Parameter, name, f)
        if name in ['add', 'sub', 'mul', 'truediv', 'pow', 'matmul']:
            setattr(Parameter, f"__{name}__", f)
            setattr(Parameter, f"__r{name}__", lambda self, x: f(x, self))
            setattr(Parameter, f"__i{name}__", lambda self, x: self.fill(f(self, x)))
        return op
    
    def __repr__(self):
        return self.__name__

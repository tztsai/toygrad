import numpy as np
import inspect
from abc import ABCMeta, ABC, abstractmethod
from utils.dev import ProfileOp, NameSpace


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
            visited.add(ctx)
            grads = ctx.passgrads(param)
            for par, grad in zip(ctx.inputs, grads):
                if isinstance(par, Parameter) and par.learnable:
                    par.grad += grad
                    dfs(par, visited)
        assert not self.shape, 'backprop must start from a scalar'
        self.grad = 1  # the gradient of the source param wrt itself is constant 1
        dfs(self)

    def __hash__(self): return id(self)
        

class OperationMeta(ABCMeta):
    """A metaclass of Parameter operations."""
    
    def __new__(meta, name, bases, namespace):
        op = super().__new__(meta, name, bases, namespace)
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
        return self.__name__ # '<Operation %s>' % self.__name__


class Operation(ABC, metaclass=OperationMeta):
    """
    The baseclass of Parameter operations.
    An instantiation of it creates a context in the computation graph.
    """
    ndim_in, ndim_out = 0, 0  # num of dims of input and output on which op is applied
    bound_axes = ()  # input-output axes that must be identical in deriv
    omitted_axes = ()  # omitted input axes in deriv
    
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
    
    def backward(self, output):
        """Computes the gradients of inputs given the output.
        Override this to define your own backprop method.
        You may want to save something for the backprop when you apply the operation -
        just add any attribute to `self` (its name better bigins with '_').
        Note that the return value should be iterable even if there is only one input.
        The default steps: (theoretically correct but probably inefficient)
        1. expand output grad: insert new axes into the output gradient ∇y
        2. swap: swap the "constrained" axes of de/dy with the corresponding new axes
        3. expand deriv: insert omitted exes into the partial derivatives dy/dx
        4. multiply: multiply ∇y with dy/dx
        5. sum: sum up the product along axes corresponding to the output to obtain ∇x
        """
        for i, x in enumerate(self.inputs):
            xdim, ydim = self.inputattr(ndim_in=i), self.ndim_out
            dy_dx = self.inputattr(deriv=i)
            bd_axs, om_axs = self.inputattr(bound_axes=i), self.inputattr(omitted_axes=i)
            off1, off2 = -xdim-ydim, -ydim  # offsets corresponding to input and output axes
            y_grad = np.expand_dims(output.grad, tuple(range(off1, off2)))
            for a in bd_axs: y_grad = np.swapaxes(y_grad, a+off1, a+off2)
            dy_dx = np.expand_dims(dy_dx, axis=tuple([a1+off1 for a1 in om_axs] +
                                                     [a2+off2 for a2 in bd_axs]))
            # basically, ∇x = Σ_y (∇y * ∂y/∂x)
            yield np.sum(y_grad * dy_dx, axis=tuple(range(off2, 0)))

    def passgrads(self, output):
        """Call the backward method and process the shape of the grads."""
        def backsplit(sh, k): return (sh[:-k], sh[-k:]) if k else (sh, ())
        for i, (x, grad) in enumerate(zip(self.inputs, self.backward(output))):
            xdim = self.inputattr(ndim_in=i)
            xsh_ext, xsh = backsplit(x.shape, xdim)
            gsh_ext, gsh = backsplit(grad.shape, xdim)
            assert gsh == xsh, 'gradient shape mismatch'
            bc_axes = []  # find broadcasted axes of x
            for i in range(len(gsh_ext)):
                k = gsh_ext[-i-1]
                try: j = xsh_ext[-i-1]
                except IndexError: j = 1
                if j == 1 and k > 1: bc_axes.append(len(gsh_ext)-i-1)
                else: assert j == k
            # debroadcast - sum over the broadcasted axes
            if bc_axes: grad = np.sum(grad, axis=tuple(bc_axes))
            yield grad.reshape(x.shape)

    def inputattr(self, **kwds):
        (a, i), = kwds.items(); val = getattr(self, a)
        return val if len(self.inputs) == 1 else val[i]

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

        self.pars = NameSpace()
        params = inspect.signature(self.apply).parameters
        for i, p in enumerate(params.values()):
            val = args[i] if p.default is p.empty else p.default
            self.pars[p.name] = val
        self.pars.update(kwds)
        
        output = Parameter(self.apply(*args, **kwds),
                            dtype=dtype, learnable=learnable)
        if learnable: output._ctx = self
        return output

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(map(repr, self.inputs))})"

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
    You can save information for backward by add new attributes to the context.
    It may be a good idea to add an underscore before the names of these new attributes.
    """
    ndim_in, ndim_out = 0, 0  # num of dims of input and output on which op is applied
    omitted_axes = ()  # omitted (independent) input axes in deriv
    bound_axes = ()
    # if i is in bound_axes, then the deriv in the i-th dim of output wrt. the i-th dim 
    # of input is non-zero only if their indices are the same
    # eg. let deriv[i,j,r,s] = dy[i,j]/dx[r,s], if 0 is in bound_axes, then deriv[i,j,r,s] != 0
    # only if i == r, so actually you only need to provide deriv[i,j,s] = dy[i,j]/dx[i,s]
    
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
    
    def backward(self, grad_out):
        """Computes the gradients of inputs ∇x given the gradient of the output ∇y.
        This default method requires you to provide self.deriv = dy/dx.
        You can also override this to directly provide the gradients of inputs.
        Note that the return value should be iterable even if there is only one input.
        """
        for i, x in enumerate(self.inputs):
            xdim, ydim = self.inputattr(ndim_in=i), self.ndim_out
            dy_dx = self.inputattr(deriv=i)
            bd_axs, om_axs = self.inputattr(bound_axes=i), self.inputattr(omitted_axes=i)
            off1, off2 = -xdim-ydim, -ydim  # offsets corresponding to input and output axes
            # insert new axes corresponding to the input dims into y_grad
            grad_y = np.expand_dims(grad_out, tuple(range(off1, off2)))
            # swap the output axes if they are bound with the corresponding input axes
            for a in bd_axs: grad_y = np.swapaxes(grad_y, a+off1, a+off2)
            # insert the missing input and output axes in the deriv
            dy_dx = np.expand_dims(dy_dx, axis=tuple([a1+off1 for a1 in om_axs] +
                                                     [a2+off2 for a2 in bd_axs]))
            # ∇x = Σ_y (∇y * ∂y/∂x)
            yield np.sum(grad_y * dy_dx, axis=tuple(range(off2, 0)))

    def passgrads(self, output):
        """Call the backward method and process the shapes of the grads."""
        def backsplit(sh, k): return (sh[:-k], sh[-k:]) if k else (sh, ())
        grads = self.backward(output.grad)
        for i, (x, grad) in enumerate(zip(self.inputs, grads)):
            xdim = self.inputattr(ndim_in=i)
            if xdim < 0: xdim = x.ndim
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
        if a in ['bound_axes', 'omitted_axes'] and not val: return ()
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

        self.pars = NameSpace()  # save all inputs in self.pars
        params = inspect.signature(self.apply).parameters
        for i, p in enumerate(params.values()):
            val = args[i] if i < len(args) else p.default
            self.pars[p.name] = val
        self.pars.update(kwds)
        
        output = Parameter(self.apply(*args, **kwds),
                            dtype=dtype, learnable=learnable)
        if learnable: output._ctx = self
        return output

    def __repr__(self):
        def rep(x): return '%s%s' % (type(x), np.shape(x)) if np.size(x) > 4 else repr(x)
        return f"{type(self).__name__}({', '.join(map(rep, self.inputs))})"

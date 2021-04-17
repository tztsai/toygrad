import numpy as np
from abc import ABCMeta, ABC, abstractmethod
from utils.dev import *


class Param(np.ndarray):
    """ A parameter in the toych model.
    There are three kinds of Param objects:
    - constant: does not take part in the autograd
    - variable: passes gradients during the autograd but cannot be trained
    - trainable: stores an array of gradients and can be trained by the optimizer
    """
    training = True
    rng = np.random.default_rng()
    grad_lim = 1e8  # magnitude limit of each element of the gradient
    kinds = dict(constant=0, variable=1, trainable=2)

    def __new__(cls, value=None, *, size=None, dtype=np.float, mean=0, scale='he',
                kind='trainable', name=None):
        """
        If `value` is given, then it will be converted to a Param.
        If `dtype` is the same as that of the given `value`, then a view of
        `value` will be returned, so its data will not be copied.
        However, if `size` is additionally specified, then a new Param
        of this size will be created filled with the given `value`.
        
        If `value` is not given, a random Param following normal
        distribution will be generated. `mean` and `scale` of the distribution
        can be specified, but if `scale` is not provided, initialization methods
        like 'he' or 'xavier' will be used to compute the scale.
        
        >>> Param([[1,2,3],[4,5,6]])
        >>> Param(size=[4, 4], dtype=np.float32, scale=1)
        >>> Param(0, size=[5, 5])
        >>> w = Param(size=[5, 3])
        >>> w is Param(w)
        """
        if value is None:  # random initialization
            if size is None: size = 1
            if type(scale) is str:
                d_in = size[0] if hasattr(size, '__len__') else size
                scale = Param.init_scale(d_in, scale)
            value = cls.rng.normal(size=size, loc=mean, scale=scale)
        else:
            if size is not None:  # fill an array of the given size
                value = np.full(size, value, dtype=dtype)
        return np.asarray(value, dtype=dtype).view(cls)
    
    @staticmethod
    def init_scale(d_in, method):
        if method == 'he':
            return np.sqrt(2/d_in)
        if method == 'xavier':
            return np.sqrt(1/d_in)
        raise NotImplementedError
    
    def __array_finalize__(self, obj):
        if not isinstance(obj, Param) or not hasattr(self, 'name'):
            self.__init__(kind='constant')
        
    def __init__(self, *args, kind='trainable', name=None, **kwds):
        self.name = name if name else type(self).__name__
        self.kind = Param.kinds.get(kind, kind)
        assert self.kind in Param.kinds.values()
        self._ctx = None
        self._grad = 0 if not self.constant else None

    @property
    def grad(self): return self._grad

    @grad.setter
    def grad(self, grad):
        if self.constant or not Param.training: return
        if np.ndim(grad) == 0:  # a scalar
            grad = np.full(self.shape, grad)
        elif np.shape(grad) != self.shape:
            raise ValueError('gradient shape mismatch')
        self._grad = np.clip(grad, -self.grad_lim, self.grad_lim)
        
    @property
    def grad_clean(self):
        return self.constant or id(self.grad) == id(0)

    def zero_grad(self):
        if not self.constant: self._grad = 0
    
    @classmethod
    @contextmanager
    def not_training(cls):
        """Use `with Param.not_training(): ...` to temporarily disable training."""
        cls.training = False; yield; cls.training = False

    @property
    def data(self): return np.asarray(self)
        
    def view(self, *args):
        if not args:
            return Param(self, dtype=self.dtype, kind=self.kind)
        else:
            return super().view(*args)
        
    def assign(self, par):
        if self.shape:
            self[:] = par; return self
        else:
            return par
        
    def backward(self):
        if not Param.training: return
        assert not self.constant and self.ndim == 0, 'backprop must start from a scalar variable'
        
        visited, params = set(), []
        def deepwalk(param):
            visited.add(param)
            if param._ctx:
                [deepwalk(p) for p in param._ctx.inputs if isinstance(p, Param)
                 and not p.constant and p not in visited]
            params.append(param)
            
        deepwalk(self)
        self.grad = np.ones(self.shape)
        for y in reversed(params):
            if (ctx := y._ctx) is None: continue
            assert not y.grad_clean
            with ProfileOp(ctx, backward=True):
                x_grads = ctx.backward(y.grad)
            for x, g in zip(ctx.inputs, x_grads):
                if isinstance(x, Param) and not x.constant:
                    x.grad += g
        # queue, params = Queue(), set()  # use breadth-first-search since grads may accumulate
        # queue.put(self)
        # while not queue.empty():
        #     y = queue.get()
        #     params.add(y)
        #     if (ctx := y._ctx) is None: continue
        #     input_grads = ctx.backward(y.grad)
        #     for x, g_x in zip(ctx.inputs, input_grads):
        #         if isinstance(x, Param) and not x.constant:
        #             x.grad += g_x
        #             if x not in params: queue.put(x)
        return params  # return visited parameters for optimization
    
    def copy(self):
        cp = Param(super().copy(), dtype=self.dtype)
        cp.__dict__.update(self.__dict__)
        return cp
    
    def __getattr__(self, name):
        if name in Param.kinds:
            return self.kind == Param.kinds[name]
        return super().__getattribute__(name)

    def __hash__(self):
        return id(self)

    def __reduce__(self):
        """Used for pickling the Param object."""
        pickled_state = super().__reduce__()
        my_state = self.__dict__.copy()
        my_state['_ctx'] = None
        my_state['_grad'] = 0 if self.trainable else None
        return (*pickled_state[:2], (*pickled_state[2], my_state))
        
    def __setstate__(self, state):
        super().__setstate__(state[:-1])
        self.__dict__.update(state[-1])
    
    def __repr__(self):
        s = f'{self.name}(#{list(self.shape)})' if self.size > 1 else super().__repr__()
        return s[:-1] + ', %s)' % next(k for k, v in Param.kinds.items() if v == self.kind)


class FunctionMeta(ABCMeta):
    def __init__(cls, name, bases, ns):
        super().__init__(name, bases, ns)
        if not inspect.isabstract(cls):
            if cls.register:
                registermethod(cls)
            if hasattr(cls, 'cache') and cls.cache:
                cls._cache = Cache()
                atexit.register(cls.log_cache_hits)

    def __call__(cls, *args, **kwds):
        if inspect.isabstract(cls):
            assert len(args) == 1 and not kwds
            # converts a function to a subclass of `cls`
            bases = (cls, *cls.__bases__)
            ns = {'apply': staticmethod(f := args[0])}
            func = FunctionMeta(f.__name__, bases, ns)
            func.__module__ = f.__module__  # for pickle to lookup attributes?
            return func
        fn = super().__call__(*args, **kwds)
        return fn
        
    def __repr__(cls):
        return cls.__name__
    
    def __reduce__(self):
        return super().__reduce__()

class AbstractFunction(ABC, metaclass=FunctionMeta):
    def __new__(cls, *args, **kwds):
        fn = object.__new__(cls)
        fn.__name__ = f"{cls.__name__}{signature_str(*args, **kwds)}"
        return fn

    @abstractmethod
    def apply(self, *args, **kwds): NotImplemented

    def update_args(self, *args, **kwds):
        return args, kwds
        
    def __call__(self, *args, **kwds):
        with ProfileOp(self):  # log elapsed time
            return self.apply(*args, **kwds)
        
    def __repr__(self):
        return self.__name__
    
    
def registermethod(fn):
    """Registers a class or a function as a method of Param, can be used as a decorator."""
    if not isinstance(fn, type): fn = Function(fn)
    def f(*args, **kwds): return fn(*args, **kwds)  # convert to a function
    setattr(Param, name := fn.__name__.lower(), f)
    if name in ['add', 'sub', 'neg', 'mul', 'truediv', 'pow', 'matmul', 'getitem']:
        setattr(Param, f"__{name}__", f)
        setattr(Param, f"__r{name}__", lambda self, x: f(x, self))
        setattr(Param, f"__i{name}__", lambda self, x: self.assign(f(self, x)))
    return fn

def wrap_call(call):
    def wrapper(self, *args, **kwds):
        if self.wait_inputs:
            return self.context(*args, **kwds)
        self.inputs = args
        output = call(self, *args, **kwds)
        return output
    return wrapper


class Function(AbstractFunction):
    """ Baseclass of functions applied to Params.
    The class is directly callable like functions.
    An instance of it acts as a node in the computation graph. 
    
    Attributes:
        blackbox: whether this appears as a single node in the compgraph
        partial: whether this function contains params to be initialized
        register: whether to register this function as a method of the Param class
    """
    blackbox = False
    register = False
    partial = False

    def __new__(cls, *args, **kwds):
        fn = super().__new__(cls, *args, **kwds)
        fn.need_init = '__init__' in cls.__dict__
        fn.parent = None  # a context may have a parent function
        return fn.decide_call(*args, **kwds)
        
    def decide_call(self, *args, **kwds):
        self.wait_inputs = ((self.partial or self.need_init) and
                            not array_at_first(args))
        return self if self.wait_inputs else self(*args, **kwds)
    
    def __init__(self, *args, **kwds):
        self.args, self.kwds = args, kwds
        self.need_init = False
    
    def context(self, *args, **kwds):
        ctx = object.__new__(type(self))
        ctx.__name__ = self.__name__ + signature_str(*args, **kwds)
        ctx.parent = self
        args, kwds = self.update_args(*args, **kwds)
        Function.__init__(ctx, *args, **kwds)
        return ctx.decide_call(*args, **kwds)

    def update_args(self, *args, **kwds):
        args += self.args
        kwds = {**self.kwds, **kwds}
        return super().update_args(*args, **kwds)
    
    def __getattr__(self, name):
        return getattr(self.parent, name)
    
    def __call__(self, *args, **kwds):
        output = super().__call__(*args, **kwds)
        if isinstance(output, Param) and self.blackbox:
            output._outer_ctx = self
        return output
    __call__ = wrap_call(__call__)
    

class Operation(Function):
    """ Baseclass of Param operations with automatic differentiation.
    You can choose to provide `self.deriv` in the `apply` method or override
    the `backward` method to directly provide the grads of inputs.
    As an example of `self.deriv`, consider apply(x, y) with x.shape=(2,3) and y.shape=(4,),
    then `self.deriv.shape` should be (2,3,4), where `self.deriv[i,j,k] = dy[k] / dx[i,j]`.
    
    Attributes:
        ndim_in, ndim_out: least num of dims of input(s) and output
    """
    register = True
    cache = True
    ndim_in, ndim_out = 0, 0
    
    def backward(self, grad_out):
        """Computes the gradients of inputs ∇x given the gradient of the output ∇y.
        You can override this to directly provide the gradients of inputs.
        Note that the return value should be iterable even if there is only one input.
        """
        for i, x in enumerate(self.inputs):
            if isinstance(x, Param) and not x.constant:  # ∂e/∂x_i = Σ_j (∂e/∂y_j * ∂y_j/∂x_i)
                xdim, ydim = ensure_list(self.ndim_in)[i], self.ndim_out
                dy_dx = ensure_list(self.deriv)[i]
                grad_y = np.expand_dims(grad_out, tuple(range(-xdim-ydim, -ydim)))
                grad = np.sum(grad_y * dy_dx, axis=tuple(range(-ydim, 0)))
                yield self.debroadcast(x, xdim, grad)
            else:
                yield None

    def debroadcast(self, input, ndim_in, grad):
        def split_tail(l, k):
            return (l[:-k], l[-k:]) if k else (l, ())
        # check the shape and debroadcast if necessary
        xsh_ext, xsh = split_tail(np.shape(input), ndim_in)
        gsh_ext, gsh = split_tail(np.shape(grad), ndim_in)
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
        return grad.reshape(np.shape(input))
    
    @classmethod
    def manage_cache(cls, key, func, args, kwds):
        if cls.cache:
            if key in (cache := cls._cache):
                return 0, cache[key]
            else:
                return 1, func(*args, **kwds), cache
        else:
            return 2, func(*args, **kwds)
        
    @classmethod
    def log_cache_hits(cls):
        dbg('%d cache hits of %s', cls._cache._cnt, cls)

    def __call__(self, *args, **kwds):
        """Wraps the apply method to process arguments and the return value."""
        binds = bind_pars(self.apply, *args, **kwds)
        
        try:
            p0 = next(p for p in binds.values() if isinstance(p, Param))
            dtype = p0.dtype
        except StopIteration:  # no Params in the arguments
            return self.apply(*args, **kwds)

        kind = 'constant'  # parameter type of the output
        def param2array(x):
            nonlocal kind
            if isinstance(x, Param):
                if not x.constant: kind = 'variable'
                return np.asarray(x)
            return x
        args = [param2array(arg) for arg in args]
        kwds = {key: param2array(val) for key, val in kwds.items()}

        for name, val in bind_pars(self.apply, *args, **kwds).items():
            setattr(self, '_'+name, val)  # store inputs for backward

        key = tuple((k, id(v)) for k, v in binds.items())
        ret = self.manage_cache(key, super(Function, self).__call__, args, kwds)
        if ret[0] == 0:  # inputs in cache
            output, other = ret[1]
            output = output.view()
            self.__dict__.update(other.__dict__)
        else:
            output = Param(ret[1], dtype=dtype, kind=kind)
            if ret[0] == 1: ret[2][key] = output, self  # save to cache
            
        output._ctx = self  # set the output as the child of the context in the computation graph
        return output
    
    __call__ = wrap_call(__call__)

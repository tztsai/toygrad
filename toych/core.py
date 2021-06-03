import numpy as np
from .utils.dev import *


class Param(np.ndarray):
    """ A parameter in the toych model.
    
    Attributes:
    - kinds:
      - constant: does not participate in the autograd
      - variable: passes gradients during the autograd but cannot be trained
      - trainable: stores an array of gradients and can be updated by the optimizer
    - rng: random generator
    - training: whether the grad of the Param can be updated
    - auto_name: auto name the Param by the variable name (unreliable!)
    - random_init: method of random initialization
    """
    kinds = dict(constant=0, variable=1, trainable=2)
    rng = np.random.default_rng()
    training = True
    auto_name = False
    random_init = 'he'

    def __new__(cls, value=None, *, size=None, dtype=None, mean=0., scale=None, **kwds):
        if type(value) is tuple:  # a tuple specifies the size instead of value
            value, size = None, value
        if value is None:  # random initialization
            assert dtype is None
            if size is None: scale = 1.
            elif scale is None: scale = cls.random_init
            if type(scale) is str:
                try: d_in = size[0]
                except: d_in = size
                scale = cls.init_scale(d_in, scale)
            value = cls.rng.normal(size=size, loc=mean, scale=scale)
        else:
            if size is not None:  # fill the value in an array of the given size
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
        
    def __init__(self, value=None, *, kind=None, name=None, **kwds):
        """ Create a Param.
        
        Args:
        - value:
        - If `value` is given, then it will be converted to a Param.
            The default `kind` in this case is "variable".
            If the first argument is a tuple, then it will be considered as `size`
            instead of `value`, i.e. `Param((...))` is equivalent to `Param(size=(...))`.
            If `dtype` is the same as that of the given `value`, then a view of
            `value` will be returned, so its data will not be copied.
            However, if `size` is additionally specified, then a new Param
            of this size will be created filled with the given `value`.
        - If `value` is not given, a random Param following normal
            distribution will be generated. `mean` and `scale` of the distribution
            can be specified, but if `scale` is not provided, initialization methods
            like 'he' or 'xavier' will be used to compute the scale.
            The default `kind` in this case is "trainable".
        - kind: 'constant' / 0, 'variable' / 1, 'trainable' / 2
        
        Examples:
        >>> Param([[1,2,3],[4,5,6]])
        >>> Param(size=[4, 4], dtype=np.float32, scale=1)
        >>> Param(0, size=[5, 5])
        >>> w = Param(size=[5, 3])
        >>> w is Param(w)
        """
        if kind is None:
            kind = 'trainable' if isinstance(value, (type(None), tuple)) else 'variable'
        self.kind = Param.kinds.get(kind, kind)
        assert self.kind in Param.kinds.values()
        self.name = name
        self._ctx = None
        self._grad = None
        
    @property
    def data(self): return np.asarray(self)

    @property
    def grad(self): return self._grad

    @grad.setter
    def grad(self, grad):
        if self.constant or not Param.training: return
        if np.shape(grad) != self.shape:
            raise ValueError('gradient shape mismatch')
        self._grad = grad
        
    @property
    def has_grad(self): return self.grad is not None

    def del_grad(self): self._grad = None

    def detach(self): self._ctx = None
    
    @classmethod
    @contextmanager
    def not_training(cls):
        cls.training = False; yield; cls.training = True
    
    def deepwalk(self):
        stack, visited, params = [[0, self]], {self}, []
        def visit(par):
            if isinstance(par, (list, tuple)):
                [visit(p) for p in par]
            elif isinstance(par, Param):
                if not par.constant and par not in visited:
                    visited.add(par); stack.append([0, par])
        while stack:  # toposort the related params
            exiting, param = stack.pop()
            if exiting:
                params.append(param)
            else:
                stack.append([1, param])
                if param._ctx:
                    [visit(p) for p in reversed(param._ctx.inputs)]
        return params

    @timeit
    def backward(self):
        if not Param.training: return
        assert not self.constant and self.ndim == 0, 'backprop must start from a scalar Param'
        params = self.deepwalk()
        self.grad = np.array(1.)
        def add_grad(x, g):  # make it recursive
            if isinstance(x, Param) and not x.constant:
                x.grad = x.grad + g if x.has_grad else g
            elif isinstance(x, (list, tuple)):
                assert len(x) == len(g)
                [add_grad(_x, _g) for _x, _g in zip(x, g)]
        for y in filter(lambda y: y._ctx, reversed(params)):
            xs, gs = y._ctx.inputs, y._ctx.backward(y.grad)
            [add_grad(x, g) for x, g in zip(xs, gs)]
        return (p for p in params if p.trainable)

    def view(self, *args):
        if not args: return Param(self, dtype=self.dtype, kind=self.kind)
        else: return super().view(*args)
    
    def copy(self):
        cp = Param(super().copy(), dtype=self.dtype)
        cp.__dict__.update(self.__dict__)
        return cp
    
    def __getattr__(self, name):
        if name in Param.kinds: return self.kind == Param.kinds[name]
        return super().__getattribute__(name)

    def __hash__(self): return id(self)
    
    def __reduce__(self):  # for serialization
        state = super().__reduce__()
        return (*state[:2], (*state[2], self.__dict__))

    def __setstate__(self, state):
        super().__setstate__(state[:-1])
        self.__dict__.update(state[-1])
    
    def __repr__(self):
        if not self.name:
            if self.auto_name:
                bindings = inspect.currentframe().f_back.f_locals
                for k, v in bindings.items():
                    if id(self) == id(v): self.name = k; break
            else:
                self.name = 'Param' if self.constant else 'P' + str(id(self))[-4:]
        s_kind = next(k for k, v in Param.kinds.items() if v == self.kind)
        s_dtype = '' if self.dtype is np.dtype('float') else ', dtype=' + self.dtype.name
        return array_repr(self).replace('[', '(<').replace(']', f'>, {s_kind}{s_dtype})')


class FunctionMeta(type):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if not isabstract(cls.apply):
            if cls.register:
                registermethod(cls)
            if '__init__' in namespace:
                cls.need_init = True

    def __call__(self, *args, **kwds):
        if isabstract(self.apply):
            assert len(args) == 1 and not kwds and callable(f := args[0])
            new_func = FunctionMeta(f.__name__, (self, *self.__bases__), {})
            new_func.apply = staticmethod(f)
            return new_func
        return super().__call__(*args, **kwds)

    def __repr__(cls):
        return cls.__name__


def registermethod(fn, name=None):
    """Registers a class or a function as a method of Param, can be used as a decorator."""
    if not isinstance(fn, type): fn = Function(fn)
    if name is None: name = fn.__name__.lower()
    setattr(Param, name, lambda *args, **kwds: fn(*args, **kwds))
    if name in ['add', 'sub', 'neg', 'mul', 'truediv', 'pow', 'matmul', 'getitem']:
        setattr(Param, f"__{name}__", lambda self, *x: fn(self, *x))
        if name not in ['neg', 'getitem']:
            setattr(Param, f"__r{name}__", lambda self, x: fn(x, self))
    return fn

def wrap_call(call):
    @wraps(call)
    def wrapper(self, *args, **kwds):
        if self.need_init:
            args, kwds = self.update_args(*args, **kwds)
            ctx = Context(self)
        else:
            ctx = self
        ctx.inputs = args
        output = call(self, *args, **kwds)
        if Param.training and isinstance(output, Param):
            if isinstance(self, Operation):
                output._ctx = ctx
            elif self.blackbox:
                output._outer_ctx = ctx
        return output
    return wrapper


class Function(metaclass=FunctionMeta):
    """ Baseclass of callable classes in toych.
    The class can be directly callable like functions (if need_init=False).
    An instance of it acts as a node in the computation graph.
    
    Attributes:
    - register: whether to register this function as a method of the Param class
    - blackbox: whether this appears as a single node in the compgraph
    - need_init: whether this function needs to be initialized
    """
    register = False
    blackbox = True
    need_init = False  # automatically set to True if the class has __init__

    def __new__(cls, *args, **kwds):
        fn = super().__new__(cls)
        fn.__sig = (cls, args, kwds)
        if array_at_first(args): fn.need_init = False
        return fn if fn.need_init else fn(*args, **kwds)
    
    def __init__(self, *args, **kwds):
        self.args, self.kwds = args, kwds
    
    @abstractmethod
    def apply(self, *args, **kwds): NotImplemented

    @wrap_call
    def __call__(self, *args, **kwds):
        return self.apply(*args, **kwds)

    def update_args(self, *args, **kwds):
        """ Update positional and keyword arguments passed to self.apply. """
        try: return args + self.args, {**self.kwds, **kwds}
        except: return args, kwds

    @property
    def __name__(self):
        if not hasattr(self, '__name'):
            cls, args, kwds = self.__sig
            self.__name = f'{cls}{signature_str(*args, **kwds)}'
        return self.__name

    def __repr__(self): return self.__name__


class Context:
    def __init__(self, fn):
        self.__fn = fn
        
    def __getattr__(self, name):
        return getattr(self.__fn, name)
    
    def __repr__(self):
        return repr(self.__fn) + signature_str(*self.inputs)
    
    def getfunc(self):
        return self.__fn

    def backward(self, grad_out):
        return type(self.__fn).backward(self, grad_out)


class Operation(Function):
    """ Baseclass of Param operations with automatic differentiation.
    You can choose to provide `self.deriv` in the `apply` method or override
    the `backward` method to directly provide the grads of inputs.
    As an example of `self.deriv`, consider apply(x, y) with x.shape=(2,3) and y.shape=(4,),
    then `self.deriv.shape` should be (2,3,4) with `self.deriv[i,j,k] = dy[k] / dx[i,j]`.
    
    Attributes:
    - ndim_in, ndim_out: least num of dims of input(s) and output
    """
    register = True
    blackbox = False
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
            else: yield None

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

    @wrap_call
    def __call__(self, *args, **kwds):
        """Wraps the apply method to process arguments and the return value."""
        def find_params(obj):
            if isinstance(obj, (list, tuple)):
                for it in obj: yield from find_params(it)
            elif isinstance(obj, Param): yield obj
        try:
            dtype = next(find_params(args + tuple(kwds.values()))).dtype
        except StopIteration:  # no Params in the arguments
            return self.apply(*args, **kwds)

        kind = 'constant'  # parameter type of the output
        def param2array(x):
            if isinstance(x, Param):
                if not x.constant:
                    nonlocal kind
                    kind = 'variable'
                return np.asarray(x)
            return x
        
        args, kwds = deepmap(param2array, (args, kwds))
        return Param(self.apply(*args, **kwds), dtype=dtype, kind=kind)


def save(obj, filename=None):
    if not filename:
        return pickle.dumps(obj)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load(filename_or_bytes):
    if type(filename_or_bytes) is bytes:
        return pickle.loads(filename_or_bytes)
    with open(filename_or_bytes, 'rb') as f:
        return pickle.load(f)

def copy(obj): return load(save(obj))

"""
Inherit from Operation to add differentiable operations (deriv or backward must be implemented).
Functions are automatically differentiable due to these basic operations.
Decorate a function using `registermethod` to register it as a method of the Param class.

Implementations of sum, max, reshape, transpose, Pool2D, BatchNorm2D, Conv2D, etc. have referred
to "tinygrad" (https://github.com/geohot/tinygrad).

There can be 3 ways to apply a function or operation:
>>> x = Param(size=[5, 3])
>>> dropout(x, 0.3)             # available to any Function
>>> x.dropout(0.3)              # if the function has been registered
>>> d = dropout(0.3); d(x)      # if the function is partial
"""
import numpy as np
from core import Param, Function, Operation, registermethod
from utils.dev import ensure_list, abstractmethod, ABC


class UnaryOp(Operation):
    ndim_in = ndim_out = 0

class exp(UnaryOp):
    def apply(self, x):
        y = np.exp(x)
        self.deriv = y
        return y

class log(UnaryOp):
    def apply(self, x):
        self.deriv = 1 / x
        return np.log(x)
    
class tanh(UnaryOp):
    def apply(self, x):
        y = np.tanh(x)
        self.deriv = 1 - y**2
        return y
    
class sign(UnaryOp):
    def apply(self, x):
        self.deriv = 0.
        return np.sign(x)

class abs(UnaryOp):
    def apply(self, x):
        self.deriv = np.sign(x)
        return np.abs(x)
    
class ReLU(UnaryOp):
    def apply(self, x):
        self.deriv = x >= 0.
        return np.maximum(x, 0.)

class dropout(UnaryOp):
    partial = True
    cache = False
    def apply(self, x, p=0.5, mask=None):
        if not Param.training: return x
        if mask is None:
            sample = np.random.rand(*np.shape(x))
            mask = (sample < 1-p) / (1-p)
        self.deriv = mask
        return mask * x


class BinaryOp(Operation):
    ndim_in, ndim_out = (0, 0), 0

class Add(BinaryOp):
    def apply(self, x, y):
        self.deriv = np.ones_like(x), np.ones_like(y)
        return x + y

class Mul(BinaryOp):
    def apply(self, x, y):
        self.deriv = y, x
        return x * y

class Pow(BinaryOp):
    def apply(self, x, y):
        self.deriv = y * x**(y-1), None #x**y * np.log(x)
        return x ** y

class maximum(BinaryOp):
    def apply(self, x, y):
        out = np.maximum(x, y)
        tx = (x == out).astype(float)
        self.deriv = tx, 1. - tx
        return out
        

class softmax(Operation):
    ndim_in, ndim_out = 1, 1
    def apply(self, x):
        ex = np.exp(x)
        y = ex / np.sum(ex, axis=-1, keepdims=True)
        I = np.eye(y.shape[-1])
        y_row, y_col = np.expand_dims(y, -2), np.expand_dims(y, -1)
        self.deriv = y_col * (I - y_row)
        return y

class smce(Operation):
    """Softmax Crossentropy"""
    ndim_in, ndim_out = (1, 1), 0
    def apply(self, x, y):
        """Note that this can only be applied when the sum of each of the rows of `y` is 1."""
        p = (ex := np.exp(x)) / np.sum(ex, axis=-1, keepdims=True)
        e = -np.sum(y * np.log(p), axis=-1)
        self.deriv = (p - y) / e.size, None
        return e.mean()


### operations that overrides the `backward` method ###

class MatMul(Operation):
    def apply(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        while x.ndim < 2: x = np.expand_dims(x, 0)
        while y.ndim < 2: y = np.expand_dims(y, -1)
        self._x, self._y = x, y
        return x @ y
    
    def backward(self, grad_out):
        grads = [grad_out @ np.swapaxes(self._y, -1, -2),
                 np.swapaxes(self._x, -1, -2) @ grad_out]
        return (g.reshape(x.shape) for x, g in zip(self.inputs, grads))

class reshape(Operation):
    def apply(self, x, shape):
        return x.reshape(shape)

    def backward(self, grad_y):
        yield grad_y.reshape(self._x.shape)

class transpose(Operation):
    def apply(self, x, order=None):
        if order is None: order = tuple(reversed(range(x.ndim)))
        self._order = order
        return np.transpose(x, order)

    def backward(self, grad_y):
        yield np.transpose(grad_y, np.argsort(self._order))

class getitem(Operation):
    def apply(self, x, idx):
        return x[idx]
    
    def backward(self, grad_y):
        grad_x = np.zeros_like(self._x)
        grad_x[self._idx] = grad_y
        yield grad_x
        
class concat(Operation):
    """Horizontally concat two arrays together."""
    def apply(self, x, y):
        self._ax = 1 if np.ndim(x) > 1 else 0
        self._i = x.shape[self._ax]
        return np.hstack([x, y])
    
    def backward(self, grad_out):
        return np.split(grad_out, [self._i], axis=self._ax)
        
def apply_to_axes(f):
    def wrapper(self, x, axis=None, keepdims=False, out=None):
        axes = range(np.ndim(x)) if axis is None else ensure_list(axis)
        for i, a in enumerate(axes):
            if a < 0: axes[i] = x.ndim + a
        return f(self, x, tuple(axes), keepdims)
    return wrapper

class sum(Operation):
    @apply_to_axes
    def apply(self, x, axes, keepdims=False):
        self._sh = [1 if i in axes else s for i, s in enumerate(np.shape(x))]
        return np.sum(x, axis=axes, keepdims=keepdims)
    
    def backward(self, grad_y):
        yield grad_y.reshape(self._sh) + np.zeros_like(self._x)

class max(Operation):
    @apply_to_axes
    def apply(self, x, axes, keepdims=False):
        y = np.max(x, axis=axes, keepdims=keepdims)
        shape = [1 if i in axes else s for i, s in enumerate(np.shape(x))]
        t = (x == y.reshape(shape))
        d = t.sum(axis=axes, keepdims=True)
        self._sh, self._d = shape, t/d
        return y
    
    def backward(self, grad_y):
        yield self._d * grad_y.reshape(self._sh)
        

### functions that are registered as Param methods ###

@registermethod
def TrueDiv(x, y): return x * (y ** -1.)

@registermethod
def Sub(x, y): return x + (-1. * y)

@registermethod
def Neg(x): return 0. - x

@registermethod
def sqrt(x): return x ** 0.5

@registermethod
def sigmoid(x): return exp(x) / (1 + exp(x))

@registermethod
def swish(x): return x * sigmoid(x)

@registermethod
def leakyReLU(x, neg_slope=0.01):
    return ReLU(x) - ReLU(neg_slope * -x)

@registermethod
def crossentropy(x, y, axis=-1, avg=True):
    e = (y * -log(x)).sum(axis=axis)
    return e.mean() if avg else e

@registermethod
def mean(x, axis=None, keepdims=False):
    s = sum(x, axis=axis, keepdims=keepdims)
    return s * np.prod(np.shape(s)) / np.prod(np.shape(x))

@registermethod
def mse(x, y, axis=None, keepdims=False):
    return mean((x - y) ** 2, axis, keepdims)

@registermethod
def var(x, axis=None, keepdims=False):
    return mse(x, mean(x, axis, keepdims=True), axis, keepdims)

@registermethod
def std(x, axis=None, keepdims=False):
    return sqrt(var(x, axis, keepdims))

@registermethod
def flatten(x):
    return reshape(x, [len(x), -1])

@registermethod
def normalize(x, axis=0):
    mu = mean(x, axis, keepdims=True)
    sigma = std(x, axis, keepdims=True)
    return (x - mu) / maximum(sigma, 1e-6)


### other functions ###
    
class Pool2D(Function):
    register = True
    partial = True
    
    def apply(self, im, size=(2,2), stride=1):
        return self.reduce(self.pool2d(im, *size, stride), axis=(-3, -1))
    
    @staticmethod
    def pool2d(im, py, px, st=1):
        (dy, ry), (dx, rx) = divmod(im.shape[-2], py*st), divmod(im.shape[-1], px*st)
        pools = im[:, :, :im.shape[-2]-ry:st, :im.shape[-1]-rx:st]
        return pools.reshape(shape=(*im.shape[:-2], dy, py, dx, px))
    
    @staticmethod
    def reduce(pools, axis):
        raise NotImplementedError

class MeanPool2D(Pool2D):
    reduce = mean
    
class MaxPool2D(Pool2D):
    reduce = max
    

### operations or methods containing parameters to be initialized ###

class Conv2D(Operation):
    def __init__(self, c_out, size, stride=1, groups=1, batchnorm=False):
        super().__init__()
        if type(size) is int: size = (size, size)
        self.c_in, self.c_out = None, c_out
        self.size, self.stride, self.groups, self.bn = size, stride, groups, batchnorm
        self.built = False

    def build(self, input):
        self.c_in = input.shape[1]
        self.filters = Param(size=[self.c_out, self.c_in, *self.size])
        self.bn = BatchNorm2D() if self.bn else None
        self.built = True
        
    def update_args(self, input):
        if not self.built: self.build(input)
        return super().update_args(input, filters=self.filters,
                                   stride=self.stride, groups=self.groups)
    
    def __call__(self, *args, **kwds):
        output = super().__call__(*args, **kwds)
        if hasattr(self, 'built') and self.built and self.bn:
            return self.bn(output)
        return output

    def apply(self, input, filters, stride=1, groups=1):
        if type(stride) is int:
            self._stride = stride = (stride, stride)
        x, w = input, filters
        cout, cin, H, W = w.shape
        ys, xs = stride
        bs = len(x)
        oy, ox = (x.shape[2]-(H-ys))//ys, (x.shape[3]-(W-xs))//xs
        assert cin * groups == x.shape[1]
        assert cout % groups == 0
        rcout = cout // groups

        gx = x.reshape(bs, groups, cin, x.shape[2], x.shape[3])
        tx = np.lib.stride_tricks.as_strided(gx,
            shape=(bs, groups, cin, oy, ox, H, W),
            strides=(*gx.strides[0:3], gx.strides[3]*ys, gx.strides[4]*xs, *gx.strides[3:5]),
            writeable=False,
        )
        tw = w.reshape(groups, rcout, cin, H, W)

        ret = np.zeros((bs, groups, oy, ox, rcout), dtype=x.dtype)
        for g in range(groups):
            # ijYXyx,kjyx -> iYXk ->ikYX
            ret[:, g] += np.tensordot(tx[:, g], tw[g], ((1, 4, 5), (1, 2, 3)))
            
        self._tx, self._tw, self._xsh = tx, tw, x.shape
        return np.moveaxis(ret, 4, 2).reshape(bs, cout, oy, ox)

    def backward(self, grad_y):
        stride, groups = self._stride, self._groups
        tx, tw, x_shape = self._tx, self._tw, self._xsh

        bs, _, oy, ox = grad_y.shape
        _, rcout, cin, H, W = tw.shape
        ys, xs = stride
        ggg = grad_y.reshape(bs, groups, rcout, oy, ox)

        gdw = np.zeros((groups, rcout, cin, H, W), dtype=tx.dtype)
        for g in range(groups):
            #'ikYX,ijYXyx -> kjyx'
            gdw[g] += np.tensordot(ggg[:, g], tx[:, g], ((0, 2, 3), (0, 2, 3)))

        # needs to be optimized
        OY, OX = x_shape[2:4]
        gdx = np.zeros((bs, groups, cin, OY, OX), dtype=tx.dtype)
        for k in range(oy*ox):
            Y, X = k//ox, k % ox
            iY, iX = Y*ys, X*xs
            #gdx[:,:,: , iY:iY+H, iX:iX+W] += np.einsum('igk,gkjyx->igjyx', ggg[:,:,:,Y,X], tw)
            for g in range(groups):
                tg = np.dot(ggg[:, g, :, Y, X].reshape(
                    bs, -1), tw[g].reshape(rcout, -1))
                gdx[:, g, :, iY:iY+H, iX:iX+W] += tg.reshape((bs, cin, H, W))

        return gdx.reshape((bs, groups*cin, OY, OX)), gdw.reshape((groups*rcout, cin, H, W))

class Affine(Function):
    def __init__(self, d_out, with_bias=True):
        super().__init__()
        self.d_out = d_out
        self.with_bias = with_bias
        self.built = False
        
    def build(self, input):
        self.d_in = input.shape[-1]
        self.w = Param(size=[self.d_in, self.d_out])
        self.b = Param(size=self.d_out) if self.with_bias else 0
        self.built = True

    def update_args(self, input):
        if not self.built: self.build(input)
        return super().update_args(input, weight=self.w, bias=self.b)

    def apply(self, input, weight, bias):
        return input @ weight + bias

class BatchNorm2D(Function):
    eps = 1e-5
    mom = 0.1  # momentum
    track_running_stats = False
    
    def __init__(self):
        super().__init__()
        self.num_batches_tracked = 0
        self.built = False
    
    @property
    def training(self):
        return Param.training

    def build(self, input):
        size = input.shape[1]
        self.weight, self.bias = Param(0, size=size), Param(0, size=size)
        self.running_mean, self.running_var = np.zeros(size), np.zeros(size)
        self.built = True
        
    def update_args(self, input):
        if not self.built: self.build(input)
        return super().update_args(input)

    def apply(self, x):
        assert self.built, 'BatchNorm2D not initialized'
        
        if self.track_running_stats or self.training:
            batch_mean = x.mean(axis=(0,2,3))
            xc = (x - batch_mean.reshape(shape=[1,-1,1,1]))
            batch_var = (xc**2).mean(axis=(0,2,3))

        if self.track_running_stats:
            self.running_mean = (1 - self.mom) * self.running_mean + self.mom * batch_mean
            self.running_var = (1 - self.mom) * self.running_var + self.mom * batch_var
            self.num_batches_tracked += 1

        if self.training:
            return self.normalize(x, batch_mean, batch_var)
        else:
            return self.normalize(x, self.running_mean, self.running_var)

    def normalize(self, x, mean, var):
        x = (x - mean.reshape(shape=[1, -1, 1, 1])) * self.weight.reshape(shape=[1, -1, 1, 1])
        return (x / (var.add(self.eps).reshape(shape=[1,-1,1,1]).sqrt()) +
                self.bias.reshape(shape=[1,-1,1,1]))
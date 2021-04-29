"""
Inherit from Operation to add differentiable operations (deriv or backward must be implemented).
Functions are automatically differentiable due to these basic operations.
Decorate a function using `registermethod` to register it as a method of the Param class.

Implementations of sum, max, reshape, transpose, Pool2D, BatchNorm2D, conv2D, etc. have referred
to "tinygrad" (https://github.com/geohot/tinygrad).

There can be 3 ways to apply a function or operation:
>>> x = Param(size=[5, 3])
>>> dropout(x, 0.3)         # available to any Function
>>> x.dropout(0.3)          # if the function has been registered
>>> dropout(0.3)(x)         # if the function's attribute 'need_init' is True
"""
import numpy as np
from .core import Param, Function, Operation, registermethod
from .utils.dev import ensure_list, random, wraps, dbg, info, timeit


class UnaryOp(Operation):
    ndim_in = ndim_out = 0

class exp(UnaryOp):
    def apply(self, x):
        y = np.exp(x)
        self.deriv = y
        return y

class log(UnaryOp):
    def apply(self, x):
        self.deriv = 1. / x
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
    
class reLU(UnaryOp):
    def apply(self, x):
        self.deriv = (x >= 0.)
        return self.deriv * x

class leakyReLU(UnaryOp):
    need_init = True
    def apply(self, x, negslope=0.01):
        self.deriv = np.maximum(negslope, x >= 0.)
        return self.deriv * x

class dropout(UnaryOp):
    need_init = True
    def apply(self, x, p=0.5, mask=None):
        if not Param.training:
            self.deriv = 1.
            return x
        if mask is None:
            sample = np.random.rand(*np.shape(x))
            mask = (sample < 1 - p) / (1 - p)
        self.deriv = mask
        return mask * x


class BinaryOp(Operation):
    ndim_in, ndim_out = (0, 0), 0

class Add(BinaryOp):
    def apply(self, x, y):
        self.deriv = np.ones_like(x), np.ones_like(y)
        return x + y
        
class Sub(BinaryOp):
    def apply(self, x, y):
        self.deriv = np.ones_like(x), np.full_like(y, -1)
        return x - y

class Mul(BinaryOp):
    def apply(self, x, y):
        self.deriv = y, x
        return x * y
    
class TrueDiv(BinaryOp):
    def apply(self, x, y):
        self.deriv = 1/y, -x/y**2 if isinstance(y, np.ndarray) else None
        return x / y

class Pow(BinaryOp):
    def apply(self, x, y):
        self.deriv = y * x**(y-1), \
            x**y * np.log(x) if isinstance(y, np.ndarray) else None
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

class softmaxCrossentropy(Operation):
    ndim_in, ndim_out = (1, 1), 0
    
    def apply(self, input, labels):
        # check whether the labels are valid (sums to 1 along the second axis)
        sample_ids = random.sample(range(len(input)), min(10, len(input)))
        assert np.allclose(np.sum(labels[sample_ids], axis=1), 1.)
        
        prs = (ex := np.exp(input)) / np.sum(ex, axis=-1, keepdims=True)
        ls = np.sum(labels * (nlls := -np.log(prs)), axis=-1)
        const_labels = not isinstance(self.inputs[1], Param) or self.inputs[1].constant
        self.deriv = (prs - labels) / ls.size, None if const_labels else nlls / ls.size
        return ls.mean()

registermethod(softmaxCrossentropy, 'smce')  # register a shorter alias


### operations that overrides the `backward` method ###

class MatMul(Operation):
    def apply(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        while x.ndim < 2: x = np.expand_dims(x, 0)
        while y.ndim < 2: y = np.expand_dims(y, -1)
        self._x, self._y = x, y
        self._xsh, self._ysh = map(np.shape, self.inputs)
        return x @ y
    
    def backward(self, grad_out):
        gx = grad_out @ np.swapaxes(self._y, -1, -2)
        gy = np.swapaxes(self._x, -1, -2) @ grad_out
        return gx.reshape(self._xsh), gy.reshape(self._ysh)

class reshape(Operation):
    def apply(self, x, *args):
        if all(type(a) is int for a in args):
            shape = args
        else:
            assert len(args) == 1
            shape = args[0]
        self._xsh = np.shape(x)
        return np.reshape(x, shape)

    def backward(self, grad_y):
        yield grad_y.reshape(self._xsh)

class transpose(Operation):
    def apply(self, x, order=None):
        if order is None: order = tuple(reversed(range(x.ndim)))
        self._order = order
        return np.transpose(x, order)

    def backward(self, grad_y):
        yield np.transpose(grad_y, np.argsort(self._order))

class getitem(Operation):
    def apply(self, x, idx):
        self._x, self._idx = x, idx
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


def convert_axis(x, axis):
    axes = range(np.ndim(x)) if axis is None else ensure_list(axis)
    for i, a in enumerate(axes):
        if a < 0: axes[i] = x.ndim + a
    return tuple(axes)

class sum(Operation):
    def apply(self, x, axis=None, keepdims=False, **_):
        axis = convert_axis(x, axis)
        self._zeros = np.zeros_like(x)
        self._sh = [1 if i in axis else s for i, s in enumerate(np.shape(x))]
        return np.sum(x, axis=axis, keepdims=keepdims)
    
    def backward(self, grad_y):
        yield grad_y.reshape(self._sh) + self._zeros

class max(Operation):
    def apply(self, x, axis=None, keepdims=False, **_):
        axis = convert_axis(x, axis)
        y = np.max(x, axis=axis, keepdims=keepdims)
        shape = [1 if i in axis else s for i, s in enumerate(np.shape(x))]
        t = (x == y.reshape(shape))
        d = t.sum(axis=axis, keepdims=True)
        self._sh, self._d = shape, t/d
        return y
    
    def backward(self, grad_y):
        yield self._d * grad_y.reshape(self._sh)


### functions that are registered as Param methods ###

@registermethod
def neg(x): return 0 - x

@registermethod
def sqrt(x): return x ** 0.5

@registermethod
def sigmoid(x): return (ex := exp(x)) / (1 + ex)

@registermethod
def swish(x): return x * sigmoid(x)

@registermethod
def crossentropy(x, y, axis=-1, avg=True):
    e = sum(y * -log(x), axis=axis)
    return mean(e) if avg else e

@registermethod
def mean(x, axis=None, keepdims=False):
    s = sum(x, axis=axis, keepdims=keepdims)
    return s * (np.prod(np.shape(s)) / np.prod(np.shape(x)))

@registermethod
def mse(x, y, axis=None, keepdims=False):
    return mean((x - y) ** 2, axis, keepdims)

@registermethod
def var(x, axis=None, keepdims=False):
    return mse(x, mean(x, axis, keepdims=True), axis, keepdims)

@registermethod
def std(x, axis=None, keepdims=False, eps=1e-5):
    return sqrt(var(x, axis, keepdims) + eps)

@registermethod
def flatten(x):
    return reshape(x, [len(x), -1])


### other functions ###

def zeros(*shape, kind='variable', dtype=float):
    return Param(np.zeros(shape, dtype=dtype), kind=kind)

def ones(*shape, kind='variable', dtype=float):
    return Param(np.ones(shape, dtype=dtype), kind=kind)


class pool2D(Function):
    register = True
    need_init = True
    reduce = NotImplemented
    
    def apply(self, im, size=(2, 2), stride=1):
        if type(size) is int: size = (size, size)
        return self.reduce(self.pool2d(im, *size, stride), axis=(-3, -1))
    
    @staticmethod
    def pool2d(im, py, px, st=1):
        (dy, ry), (dx, rx) = divmod(im.shape[-2], py*st), divmod(im.shape[-1], px*st)
        pools = im[:, :, :im.shape[-2]-ry:st, :im.shape[-1]-rx:st]
        return pools.reshape(*im.shape[:-2], dy, py, dx, px)

class meanPool(pool2D):
    reduce = mean
    
class maxPool(pool2D):
    reduce = max
    

### operations or methods containing parameters to be initialized ###

class affine(Function):
    def __init__(self, d_out, with_bias=True):
        self.d_out = d_out
        self.with_bias = with_bias
        self.built = False
        
    def build(self, input):
        self.d_in = input.shape[-1]
        self.w = Param(size=[self.d_in, self.d_out])
        self.b = Param(size=self.d_out) if self.with_bias else None
        self.built = True
        dbg(f'init affine: in_dims={self.d_in}, out_dims={self.d_out}')

    def update_args(self, input):
        if not self.built: self.build(input)
        return super().update_args(input, weight=self.w, bias=self.b)

    def apply(self, input, weight, bias=None):
        return input @ weight if bias is None else input @ weight + bias

class conv2D(Operation):
    """Convolve 2D images with filters."""
    
    def __init__(self, c_out, size, stride=1, groups=1):
        if type(size) is int: size = (size, size)
        self.c_in, self.c_out = None, c_out
        self.size, self.stride, self.groups = size, stride, groups
        self.built = False

    def build(self, input):
        self.c_in = input.shape[1]
        self.filters = Param(size=[self.c_out, self.c_in, *self.size])
        self.built = True
        dbg('init conv2D: im_size=%s, in_channels=%d, out_channels=%d',
            input.shape[-2:], self.c_in, self.c_out)
        
    def update_args(self, input):  # returns the args passed to "self.apply"
        if not self.built: self.build(input)
        return super().update_args(input, filters=self.filters,
                                   stride=self.stride, groups=self.groups)
    
    def apply(self, input, filters, stride=1, groups=1):
        if type(stride) is int:
            self._stride = stride = (stride, stride)
        self._groups = groups

        bs, ims, ih, iw = input.shape
        c_out, c_in, fh, fw = filters.shape
        c_out //= groups
        ys, xs = stride
        oh, ow = (ih - fh) // ys + 1, (iw - fw) // xs + 1
        assert ims == c_in * groups
        assert c_out * groups == filters.shape[0]

        tf = filters.reshape(groups, c_out, c_in, fh, fw)
        gx = input.reshape(bs, groups, c_in, ih, iw)
        tx = np.lib.stride_tricks.as_strided(gx,
            shape=(bs, groups, c_in, oh, ow, fh, fw),
            strides=(*gx.strides[:3], gx.strides[3]*ys, gx.strides[4]*xs, *gx.strides[3:]),
            writeable=False,
        )

        out = np.zeros((bs, groups, oh, ow, c_out), dtype=input.dtype)
        for g in range(groups):
            out[:, g] += np.tensordot(tx[:, g], tf[g], ((1, 4, 5), (1, 2, 3)))
            
        self._tx, self._tf, self._xsh, self._fsh = tx, tf, input.shape, filters.shape
        return np.moveaxis(out, 4, 2).reshape(bs, -1, oh, ow)

    def backward(self, grad_out):
        tx, tf = self._tx, self._tf
        bs, _, ih, iw = self._xsh
        _, _, oh, ow = grad_out.shape
        groups, c_out, c_in, fh, fw = tf.shape
        ys, xs = self._stride
        groups = self._groups
        
        gy = grad_out.reshape(bs, groups, c_out, oh, ow)
        gf = np.zeros(tf.shape)
        # gf = np.zeros((groups, c_out, c_in, fh, fw), dtype=tx.dtype)
        for g in range(groups):
            gf[g] += np.tensordot(gy[:, g], tx[:, g], ((0, 2, 3), (0, 2, 3)))
        gf = gf.reshape(self._fsh)

        if isinstance(ims := self.inputs[0], Param) and not ims.constant:
            # needs to be optimized
            gx = np.zeros((bs, groups, c_in, ih, iw))
            for k in range(oh * ow):
                i, j = divmod(k, ow)
                si, sj = i * ys, j * xs
                #gdx[:,:,: , iY:iY+H, iX:iX+W] += np.einsum('igk,gkjyx->igjyx', ggg[:,:,:,Y,X], tw)
                for g in range(groups):
                    tg = np.dot(gy[:, g, :, i, j].reshape(bs, -1), tf[g].reshape(c_out, -1))
                    gx[:, g, :, si:si+fh, sj:sj+fw] += tg.reshape((bs, c_in, fh, fw))
            gx = gx.reshape(self._xsh)
        else: gx = None
            
        return gx, gf

class normalize(Function):
    register = True
    eps = 1e-5
    mom = 0.9
    axis = 0  # the axis along which to apply normalization
    
    def __init__(self, axis=None, track_stats=False):
        if axis is not None:  # otherwise use class attribute
            self.axis = axis
        self.track_len = 0
        self.track_stats = track_stats
        self.built = False
    
    def build(self, input):
        axes = ensure_list(self.axis)
        shape = [1 if i in axes else s for i, s in enumerate(np.shape(input))]
        self.w = Param(size=shape)
        self.b = Param(size=shape)
        self.running_mean = np.zeros(shape)
        self.running_std = np.zeros(shape)
        self.built = True
    
    def update_args(self, input):
        if not self.built: self.build(input)
        return super().update_args(input)
    
    def apply(self, input, axis=None):
        if not hasattr(self, 'built'):
            self.built = False
        else:
            assert axis is None
        if axis is None:
            axis = self.axis

        batch_mean = mean(input, axis, keepdims=True)
        batch_std = std(input, axis, keepdims=True, eps=self.eps)

        if self.built and self.track_stats:
            m = 0. if self.track_len == 0 else self.mom
            self.track_len += 1
            self.running_mean[:] = m * self.running_mean + (1 - m) * batch_mean
            self.running_std[:]  = m * self.running_std + (1 - m) * batch_std
            x = (input - self.running_mean) / self.running_std
        else:
            x = (input - batch_mean) / batch_std

        if self.built and Param.training:
            return x * self.w + self.b
        else:
            return x

class normalize2D(normalize):
    axis = (0, 2, 3)

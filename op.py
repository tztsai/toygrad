import numpy as np
from core import Operation


def UnaryOp(op):
    op.ndim_in, op.ndim_out = 0, 0
    op.omitted_axes = op.bound_axes = ()
    return op

@UnaryOp
class ReLU(Operation):
    def apply(self, x):
        self.deriv = x >= 0
        return np.maximum(x, 0)

@UnaryOp
class Log(Operation):
    def apply(self, x):
        self.deriv = 1 / x
        return np.log(x)

@UnaryOp
class Exp(Operation):
    def apply(self, x):
        y = np.exp(x)
        self.deriv = y
        return y
    
@UnaryOp
class Tanh(Operation):
    def apply(self, x):
        y = np.tanh(x)
        self.deriv = 1 - y**2
        return y


Function = object  # TODO: modify the classes inheriting from Function


def BinaryOp(op):
    op.ndim_in, op.ndim_out = (0, 0), 0
    op.omitted_axes = op.bound_axes = ((), ())
    return op

@BinaryOp
class Add(Operation):
    def apply(self, x, y):
        self.deriv = np.ones_like(x), np.ones_like(y)
        return x + y

@BinaryOp
class Sub(Operation):
    def apply(self, x, y):
        self.deriv = np.ones_like(x), -np.ones_like(y)
        return x + y

@BinaryOp
class Mul(Operation):
    def apply(self, x, y):
        self.deriv = y, x
        return x * y
    
@BinaryOp
class TrueDiv(Operation):
    def apply(self, x, y):
        self.deriv = 1/y, -x/y**2
        return x / y

@BinaryOp
class Pow(Operation):
    def apply(self, x, y):
        self.deriv = y * x**(y-1), x**y * np.log(x)
        return x ** y
    

def AxesOp(op):
    apply = op.apply
    def apply_(self, x, axis=None):
        # if axis is None: axis = tuple(range(x.ndim))
        # if type(axis) is int: axis = [axis]
        # self.bound_axes = [a for a in range(x.ndim) if a in axis]
        y = apply(self, x, axis=axis)
        self.ndim_in, self.ndim_out = x.ndim, y.ndim
        return y
    op.apply = apply_
    return op

@AxesOp
class Reshape(Operation):
    def apply(self, x, shape):
        return x.reshape(shape)

    def backward(self, y):
        yield y.grad.reshape(self.pars.shape)

@AxesOp
class Transpose(Operation):
    def forward(self, x, order):
        return np.transpose(x, order)

    def backward(self, y):
        yield np.transpose(y, np.argsort(self.pars.order))

def inner_slice(x, arg):
    padding = [(max(0, -p[0]), max(0, p[1]-x.shape[i]))
               for i, p in enumerate(arg)]
    x = np.pad(x, padding)
    slicee = [(p[0] + padding[i][0], p[1] + padding[i][0])
              for i, p in enumerate(arg)]
    return x[[slice(x[0], x[1], None) for x in slicee]]

@AxesOp
class Slice(Function):
    def apply(self, x, arg=None):
        self._shape = x.shape
        return inner_slice(x, arg)

    def backward(self, output):
        return inner_slice(
            output.grad, 
            [(-p[0], output.grad.shape[i] + (self._shape[i]-p[1]))
             for i, p in enumerate(self.pars.arg)])
    
@AxesOp
class Sum(Operation):
    def apply(self, x, axis=None):
        return np.sum(x, axis=axis)
    
    def backward(self, output):
        x = self.pars.x
        axis = [axis] if type(axis := self.pars.axis) is int else axis
        shape = [1 if axis is None or i in axis else x.shape[i]
                 for i in range(len(x.shape))]
        yield output.grad.reshape(shape) + np.zeros_like(x)
        
# @AxesOp
class Max(Function):
    def apply(self, x, axis=None):
        axis = [axis] if type(axis) == int else axis
        ret = np.amax(x, axis=None if axis is None else tuple(
            axis), keepdims=True)

        shape = [1 if axis is None or i in axis else input.shape[i]
                 for i in range(len(input.shape))]
        ret2 = (input == ret.reshape(shape))
        div = ret2.sum(
            axis=None if axis is None else tuple(axis), keepdims=True)
        self.deriv = ret2 / div  # TODO: check correctness

        if axis is not None:
            ret = ret.reshape([x.shape[i]
                               for i in range(len(x.shape)) if i not in axis])
        return ret
    
    
##### 1D operations #####

class SoftMax(Operation):
    ndim_in, ndim_out = 1, 1

    def apply(self, x):
        ex = np.exp(x)
        y = ex / np.sum(ex, axis=-1)
        I = np.eye(y.shape[-1])
        y_row, y_col = np.expand_dims(y, -2), np.expand_dims(y, -1)
        self.deriv = y_col * (I - y_row)
        return y
    
class CrossEntropy(Operation):
    ndim_in, ndim_out = (1, 1), 0
    
    def apply(self, x, t):
        y = -np.sum(t * np.log(x), axis=-1)
        self.deriv = -t / x
        return y

class SoftMaxCrossEntropy(Operation):
    ndim_in, ndim_out = (1, 1), 0

    def apply(self, x, t):
        ex = np.exp(x)
        p = ex / np.sum(ex, axis=-1, keepdims=True)
        y = -np.sum(t * np.log(p), axis=-1)
        self.deriv = p - t
        return y


##### 2D operations ######

class MatMul(Operation):
    ndim_in, ndim_out = (2, 2), 2
    omitted_axes = [0], [1]
    bound_axes = [0], [1]
    
    def apply(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        self._xsh, self._ysh = x.shape, y.shape
        while x.ndim < 2: x = np.expand_dims(x, 0)
        while y.ndim < 2: y = np.expand_dims(y, -1)
        self.deriv = y, np.swapaxes(x, -1, -2)
        return x @ y
    
    def passgrads(self, output):
        grad_x, grad_y = super().passgrads(output)
        return grad_x.reshape(self._xsh), grad_y.reshape(self._ysh)

class Conv2D(Function):
    @staticmethod
    def forward(ctx, x, w, stride=1, groups=1):
        if type(ctx.stride) == int:
            ctx.stride = (ctx.stride, ctx.stride)
        cout, cin, H, W = w.shape
        ys, xs = ctx.stride
        bs, cin_ = x.shape[0], x.shape[1]
        oy, ox = (x.shape[2]-(H-ys))//ys, (x.shape[3]-(W-xs))//xs
        assert cin*ctx.groups == cin_
        assert cout % ctx.groups == 0
        rcout = cout//ctx.groups

        gx = x.reshape(bs, ctx.groups, cin, x.shape[2], x.shape[3])
        tx = np.lib.stride_tricks.as_strided(gx,
            shape=(bs, ctx.groups,
                   cin, oy, ox, H, W),
            strides=(*gx.strides[0:3], gx.strides[3]*ys, gx.strides[4]*xs, *gx.strides[3:5]),
            writeable=False,
        )
        tw = w.reshape(ctx.groups, rcout, cin, H, W)
        ctx.save_for_backward(tx, tw, x.shape)

        ret = np.zeros((bs, ctx.groups, oy, ox, rcout), dtype=x.dtype)
        for g in range(ctx.groups):
            # ijYXyx,kjyx -> iYXk ->ikYX
            ret[:, g] += np.tensordot(tx[:, g], tw[g], ((1, 4, 5), (1, 2, 3)))
        return np.moveaxis(ret, 4, 2).reshape(bs, cout, oy, ox)

    @staticmethod
    def backward(ctx, deriv_output):
        bs, _, oy, ox = deriv_output.shape
        tx, tw, x_shape = ctx.saved_tensors
        _, rcout, cin, H, W = tw.shape
        ys, xs = ctx.stride
        OY, OX = x_shape[2:4]

        ggg = deriv_output.reshape(bs, ctx.groups, rcout, oy, ox)

        gdw = np.zeros((ctx.groups, rcout, cin, H, W), dtype=tx.dtype)
        for g in range(ctx.groups):
            #'ikYX,ijYXyx -> kjyx'
            gdw[g] += np.tensordot(ggg[:, g], tx[:, g], ((0, 2, 3), (0, 2, 3)))

        # needs to be optimized
        gdx = np.zeros((bs, ctx.groups, cin, OY, OX), dtype=tx.dtype)
        for k in range(oy*ox):
            Y, X = k//ox, k % ox
            iY, iX = Y*ys, X*xs
            #gdx[:,:,: , iY:iY+H, iX:iX+W] += np.einsum('igk,gkjyx->igjyx', ggg[:,:,:,Y,X], tw)
            for g in range(ctx.groups):
                tg = np.dot(ggg[:, g, :, Y, X].reshape(
                    bs, -1), tw[g].reshape(rcout, -1))
                gdx[:, g, :, iY:iY+H, iX:iX+W] += tg.reshape((bs, cin, H, W))

        return gdx.reshape((bs, ctx.groups*cin, OY, OX)), gdw.reshape((ctx.groups*rcout, cin, H, W))


# @main
def test():
    from core import Parameter
    from utils.graph import show_compgraph, label
    A = Parameter(size=[100, 20, 1, 30])
    B = Parameter(size=[100, 1, 50, 30])
    C = (A * B).relu() / B.sum()
    D = Parameter(size=[30, 40])
    E = C @ D
    F = E.exp().sum()
    F.backward()
    [label(p, s) for s, p in locals().items() if isinstance(p, Parameter)]
    return show_compgraph(F, 'dot')

test()

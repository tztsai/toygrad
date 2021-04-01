import numpy as np
from core import Operation
from my_utils.utils import main, interact


def ScalarOp(op):
    ndim_in, ndim_out = 0, 0
    bound_axes, omitted_axes = (), ()
    return op

@ScalarOp
class ReLU(metaclass=Operation):
    def apply(self, x):
        self.deriv = x >= 0
        return np.maximum(x, 0)

@ScalarOp
class Log(metaclass=Operation):
    def apply(self, x):
        self.deriv = 1 / x
        return np.log(x)

@ScalarOp
class Exp(metaclass=Operation):
    def apply(self, x):
        y = np.exp(x)
        self.deriv = y
        return y


def ReduceOp(op):
    apply = op.apply
    def apply_(self, x, axis=None):
        if axis is not None:
            if not np.shape(axis): axis = [axis]
            self.bound_axes = tuple(-1 if a not in axis else a
                                    for a in range(x.ndim))
        y = apply(self, x, axis=axis)
        self.ndim_in, self.ndim_out = x.ndim, y.ndim
        return y
    op.apply = apply_
    return op

@ReduceOp
class Sum(metaclass=Operation):
    def apply(self, x, axis=None):
        self.deriv = np.ones_like(x)  #FIXME: this only applies to axis=None
        return np.sum(x, axis=axis)

@ReduceOp
class Max(metaclass=Operation):
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


Function = object  # TODO: modify the classes inheriting from Function

@ScalarOp
class Add(metaclass=Operation):
    def apply(self, x, y):
        self.deriv = np.ones_like(x), np.ones_like(y)
        return x + y

@ScalarOp
class Sub(Function):
    def apply(self, x, y):
        self.deriv = np.ones_like(x), -np.ones_like(y)
        return x + y

@ScalarOp
class Mul(metaclass=Operation):
    def apply(self, x, y):
        # self.deriv = (unbroadcast(y*deriv_output, x.shape),
        #              unbroadcast(x*deriv_output, y.shape))
        self.deriv = y, x
        return x * y

@ScalarOp
class Pow(metaclass=Operation):
    def apply(self, x, y):
        # self.deriv = (unbroadcast(y * (x**(y-1.0)) * deriv_output, x.shape),
        #              unbroadcast((x**y) * np.log(x) * deriv_output, y.shape))
        self.deriv = y * x**(y-1), x**y * np.log(x)
        return x ** y


class Reshape(Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.save_for_backward(x.shape)
        return x.reshape(shape)

    @staticmethod
    def backward(ctx, deriv_output):
        in_shape, = ctx.saved_tensors
        return deriv_output.reshape(in_shape)

class Transpose(Function):
    @staticmethod
    def forward(ctx, x, order):
        ctx.save_for_backward(order)
        return np.transpose(x, order)

    @staticmethod
    def backward(ctx, x):
        return np.transpose(x, np.argsort(ctx.order))

def inner_slice(x, arg):
    padding = [(max(0, -p[0]), max(0, p[1]-x.shape[i]))
               for i, p in enumerate(arg)]
    x = np.pad(x, padding)
    slicee = [(p[0] + padding[i][0], p[1] + padding[i][0])
              for i, p in enumerate(arg)]
    return x[[slice(x[0], x[1], None) for x in slicee]]

class Slice(Function):
    @staticmethod
    def forward(ctx, x, arg=None):
        ctx.save_for_backward(x.shape)
        return inner_slice(x, arg)

    @staticmethod
    def backward(ctx, deriv_output):
        shape, = ctx.saved_tensors
        narg = [(0-p[0], deriv_output.shape[i]+(shape[i]-p[1]))
                for i, p in enumerate(ctx.arg)]
        return inner_slice(deriv_output, narg)


class MatMul(metaclass=Operation):
    ndim_in, ndim_out = (2, 2), 2
    bound_axes = (0, -1), (-1, 1)
    omitted_axes = (0,), (1,)
    
    def apply(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        self._x_sh, self._y_sh = x.shape, y.shape
        while x.ndim < 2: x = np.expand_dims(x, 0)
        while y.ndim < 2: y = np.expand_dims(y, -1)
        self.deriv = y, np.swapaxes(x, -1, -2)
        return x @ y
    
    def backward(self, child):
        grad_x, grad_y = super().backward(child)
        return grad_x.reshape(self._x_sh), grad_y.reshape(self._y_sh)

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
    from utils.graph import show_compgraph, LABELS
    # A = Parameter([1, 2, 3])
    # B = Parameter([4, -2, 1])
    # C = A + B
    C = Parameter([[[2,5,3],[6,-1,12]]] * 100)
    D = Parameter([[2,5],[4,1],[7,4]])
    E = C @ D
    F = E.sum()
    F.backward()
    print(C.grad); print(D.grad)
    LABELS.update((v, k) for k, v in locals().items() if len(k) == 1)
    show_compgraph(E)
    # x = Parameter(size=[10, 3])
    # w = Parameter(size=[3, 2])

test()

import itertools
from importer import *

Param.rng = np.random
np.random.seed(0)
random.seed(0)
# setloglevel('debug')


def checkgrad(w, lossfun):
    def numgrad(w, h=1e-6):
        grad = np.zeros(w.shape)
        for idx in itertools.product(*map(range, w.shape)):
            w1 = w.copy()
            w1[idx] += h
            Lw1 = lossfun(w1)
            w1[idx] -= h * 2
            Lw2 = lossfun(w1)
            grad[idx] = (Lw1 - Lw2) / (2 * h)
        return grad
    
    Lw = lossfun(w)
    params = Lw.backward()
    g1, g2 = w.grad, numgrad(w)
    for p in params: p.del_grad()
    
    if not np.allclose(g1, g2, rtol=1e-3, atol=1e-6):
        graph.show_graph(Lw)
        raise AssertionError(str(np.abs(g1-g2).max()))
    

x = Param(size=[50, 3])
x1 = Param((2, 3))
x2 = Param(size=[256, 16])
x3 = Param((2, 16))
w = Param(size=[3, 5])
w2 = Param(size=[5, 5])
w3 = Param(size=[5, 4])
w4 = Param(size=[3, 24])
w5 = Param(size=[16, 8])
w6 = Param(size=[1, 8])
b = Param(size=5)
y = Param(size=[50, 5])
y2 = Param(size=[2, 10], kind='constant').softmax()
t = onehot(np.random.randint(5, size=50), k=5)
t1 = Param(onehot([2, 3], 5))
v = Param(size=[50, 5])
im = Param(size=[2, 3, 50, 50])
im2 = Param(size=[1, 2, 8, 8])
k = Param(size=[3, 3, 3, 3])
k1 = Param(size=[8, 3, 4, 4])
fc = Param(size=[3528, 10])
L = affine(5); L(x)
a = onehot(np.random.randint(5, size=50), 5).astype(bool)
A1 = affine(24)
A2 = affine(5)
N = normalize()
A2(N(A1(x1)))
C = conv2D(3, size=2); C(im2)


def fixed_dropout(p=0.5):
    mask = None
    def apply(x):
        nonlocal mask
        if mask is None:
            ret = dropout(x, p)
            mask = ret._ctx.deriv
        else:
            ret = dropout(x, mask=mask)
        return ret
    return apply

model = Compose(
    affine(24), #normalize(),
    tanh, fixed_dropout(0.4),
    affine(5), softmax
); model(x1)

model2 = Compose(
    affine(8), #normalize(),
    leakyReLU, #fixed_dropout(),
    affine(64), #normalize(),
    leakyReLU, #fixed_dropout(),
    affine(5)
); model2(x3)


for line in '''
w: (x @ w).sigmoid().sum()
w: (x @ w).swish().sum()
x1: (x1 @ w + (2*x1) @ w).softmax().crossentropy(t1)
x1: (model(x1) + model(x1*2)).sum() #crossentropy(t1)
x1: model(x1).log().mean()
x3: model2(x3).mean()
x1: N(A1(x1)).mean()
x1: A2(N(A1(x1)).relu()).softmax().mean()
w: w.normalize().mean()
w: ((x @ w).normalize() @ w2).mse(y)
w: w.leakyrelu().sum()
w: ((x @ w).exp() / (1 + (x @ w).exp())).sum()
w: (x @ w).leakyrelu().sum()
w: (x @ w).max(axis=1).sum()
w: (w.maximum(w2[:3])).mean()
w2: (w * w.maximum(w2[:3])).mean()
w: w.softmax().log().mean()
w2: w2.concat(w3).tanh().sum()
w3: w2.concat(w3).tanh().sum()
w: ((x @ w + b) @ w2 + b).mse(y)
w: affine(x, w, b).tanh().mean()
v: v.exp().log().sum()
v: v.softmax().crossentropy(y)
v: v.smce(t)
v: (v.exp() + v.exp()).sum()
v: v.sigmoid().mean()
v: (v.exp() / (1 + v.exp())).mean()
v: (v**5).mse(y)
v: v.mse(y)
v: v.tanh().mse(y)
v: v.exp().mse(y)
w: (x @ w).mean()
w: (x @ w).sum()
w: w[:2, :2].sum()
w: (x @ w + 1).mean(axis=0).sum()
w: (x @ w + b).smce(t)
w: [setattr(L, 'w', w), L(x).mse(y)][1]
b: (x @ w + b).tanh().sum()
k: im.conv2d(k).maxpool().mean()
k: im.conv2d(k[:2, :, :2, :2]).mean()
k: im.conv2d(k, stride=2).reshape([4, -1]).sum()
k: im.conv2d(k, stride=3).conv2d(k).mean()
k: (im.conv2d(k, stride=2).conv2d(k1).reshape([2, -1]) @ fc).smce(y2)
im2: C(im2).normalize2d().maxpool().mean()
'''.strip().splitlines():
    par, expr = map(str.strip, line.split(':', 1))
    if par.startswith('#'): continue
    print('checking grad of "%s"' % par, 'wrt\n ', expr)
    cost = eval(f'lambda {par}: eval(expr)')
    checkgrad(eval(par), cost)
    print('OK!')

from init import *
import itertools
from my_utils.utils import main

Param.rng = np.random
np.random.seed(0)
random.seed(0)

def checkgrad(w, lossfun):
    def numgrad(w, h=1e-5):
        grad = np.zeros(w.shape)
        for idx in itertools.product(*map(range, w.shape)):
            w1 = w.copy()
            w1[idx] += h
            Lw1 = lossfun(w1)
            grad[idx] = (Lw1 - Lw) / h
        return grad
    
    Lw = lossfun(w)
    # graph.show_compgraph(Lw)
    params = Lw.backward()#debugfile='autograd')
    g1, g2 = w.grad, numgrad(w)
    for p in params: p.zero_grad()
    assert np.allclose(g1, g2, rtol=1e-3, atol=1e-6), '%f' % np.abs(g1-g2).max()
    
    
setloglevel('INFO')

if __name__ == '__main__':
    x = Param(size=[50, 3], kind=0)
    y = Param(size=[50, 5], kind=0)
    t = np.random.randint(5, size=50)
    t = onehot(t, k=5)
    v = Param(size=[50, 5])
    w = Param(size=[3, 5])
    w2 = Param(size=[5, 5])
    w3 = Param(size=[5, 4])
    b = Param(size=5)
    im = Param(size=[2, 3, 50, 50], kind=0)
    y2 = Param(size=[2, 10], kind=0).softmax()
    k = Param(size=[4, 3, 3, 3])
    fc = Param(size=[2304, 10])
    L = Affine(5); L(x)
    C = Conv2D(4, 3, stride=2); C(im)
    w4 = Param(size=[3, 24])
    x2 = Param(size=[256, 16])
    w5 = Param(size=[16, 8])
    w6 = Param(size=[1, 8])
    a = np.random.randint(5, size=50)
    oha = onehot(a, 5).astype(bool)

    def loss1(y):
        return y.crossentropy(t)

    def loss2(y):
        return -y[oha].log().mean()

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
        Affine(24), normalize(),
        tanh, fixed_dropout(0.4),
        Affine(5), softmax
    ); model(x)

    model2 = Compose(
        Affine(8), normalize(),
        leakyReLU, #fixed_dropout(),
        Affine(64), normalize(),
        leakyReLU, #fixed_dropout(),
        Affine(5)
    ); model2(x)
    
    fixed_dropout = fixed_dropout()

    for line in '''
    w: (x @ w).sigmoid().sum()
    w: ((x @ w).normalize() @ w2).mse(y)
    w5: [setattr(model2[0], 'w', w5), model2(x2).mean()][1]
    w6: [setattr(model2[1], 'w', w6), model2(x2).mean()][1]
    w: w.normalize().mean()
    w: w.leakyrelu().sum()
    w4: [setattr(model[0], 'w', w4), loss1(model(x) + model(2*x))][1]
    w4: [setattr(model[0], 'w', w4), loss2(model(x))][1]
    w: ((x @ w).exp() / (1 + (x @ w).exp())).sum()
    w: (x @ w).leakyrelu().sum()
    w: (x @ w).max(axis=1).sum()
    w: (w.maximum(w2[:3])).mean()
    w2: (w * w.maximum(w2[:3])).mean()
    w: (fixed_dropout(w) + fixed_dropout(w*2)).sum()
    w: w.softmax().log().mean()
    w2: w2.concat(w3).tanh().sum()
    w3: w2.concat(w3).tanh().sum()
    w: ((x @ w + b) @ w2 + b).mse(y)
    w: Affine(x, w, b).tanh().mean()
    w: v.exp().log().sum()
    w: v.softmax().crossentropy(y)
    w: v.smce(t)
    w: (v.exp() + v.exp()).sum()
    w: v.sigmoid().mean()
    w: (v.exp() / (1 + v.exp())).mean()
    w: (v**5).mse(y)
    w: v.mse(y)
    w: v.tanh().mse(y)
    w: v.exp().mse(y)
    w: (x @ w).mean()
    w: (x @ w).sum()
    w: w[:2, :2].sum()
    w: (x @ w + 8).mean(axis=0).sum()
    w: (x @ w + b).smce(t)
    w: [setattr(L, 'w', w), L(x).mse(y)][1]
    b: (x @ w + b).sum()
    b: (x @ w + b).tanh().sum()
    w: (x @ w + b).softmax().crossentropy(t)
    k: im.conv2d(k).maxpool2d().mean()
    k: im.conv2d(k[:2, :, :2, :2]).mean()
    k: im.conv2d(k).reshape([4, -1]).sum()
    k: (im.conv2d(k, stride=2).reshape([2, -1]) @ fc).smce(y2)
    k: [setattr(C, 'filters', k), C(im).flatten().mean()][1]
    '''.strip().splitlines():
        par, exp = line.split(':', 1)
        print('checking grad of "%s"' % par.strip(), 'wrt\n ', exp.strip())
        cost = eval(f'lambda {par}: eval(exp)')
        checkgrad(eval(par), cost)
        print('OK!')

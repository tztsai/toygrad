from init import *
import itertools


def checkgrad(w, lossfun):
    def numgrad(w, h=1e-6):
        grad = np.zeros(w.shape)
        for idx in itertools.product(*map(range, w.shape)):
            w1 = w.copy()
            w1[idx] += h
            Lw1 = lossfun(w1)
            grad[idx] = (Lw1 - Lw) / h
        return grad
    
    Lw = lossfun(w)
    params = Lw.backward()
    g1, g2 = w.grad, numgrad(w)
    for p in params: p.zero_grad()
    assert np.allclose(g1, g2, atol=1e-6)


im = Param(size=[2, 3, 50, 50], kind=0)
y = Param(size=[2, 10], kind=0).softmax()
k = Param(size=[4, 3, 3, 3])
# fc = Affine(10)
lin = Param(size=[9216, 10])

# fc(im.conv2d(k).reshape([2, -1])).smce(y)
for exp in '''
im.conv2d(k).mean()
im.conv2d(k[:2, :, :2, :2]).mean()
im.conv2d(k).reshape([4, -1]).sum()
(im.conv2d(k, stride=2).reshape([2, -1]) @ lin).smce(y)
'''.strip().splitlines():
    print(exp, '===>')
    def loss(k): return eval(exp)
    checkgrad(k, loss)
    print('OK!')


x = Param(size=[50, 3], kind=0)
y = Param(size=[50, 5], kind=0)
t = np.random.randint(5, size=50)
t = onehot(t, k=5)
v = Param(size=[50, 5])
w = Param(size=[3, 5])
w2 = Param(size=[5, 5])
b = Param(size=5)

#%%
for exp in '''
((x @ w + b) @ w2).mse(y)
Affine(x, w, b).tanh().mean()
v.exp().log().sum()
v.softmax().crossentropy(y)
v.smce(t)
(v.exp() + v.exp()).sum()
v.sigmoid().mean()
(v.exp() / (1 + v.exp())).mean()
(v**5).mse(y)
v.mse(y)
v.tanh().mse(y)
v.exp().mse(y)
(x @ w).mean()
(x @ w).sum()
w[:2, :2].sum()
(x + 8).mean(axis=0).sum()
(x @ w + b).smce(t)
(x @ w + b).sum()
(x @ w + b).tanh().sum()
(x @ w + b).softmax().crossentropy(t)
'''.strip().splitlines():
    print(exp, '===>') 
    loss = lambda w: eval(exp)
    checkgrad(w, loss)
    print('OK!')
    

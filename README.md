# Toych

State-of-the-art deep learning framework. Just kidding 🤣🤣

## How to play

Try out the following code.

```python
>>> import toych as tc
>>> from toych import Param

"Use toych operations or functions like numpy functions."
>>> tc.max([1, 2, 3])
3
>>> tc.sum(np.array([1, 2, 3]))
6

"Create a Param. There are 3 kinds of Param: constant, variable and trainable."
>>> Param.auto_name = True  # automatically name the Param (only for fun, can be quite slow)
>>> x = Param([-1, 0, 1]); x
x(<3>, variable, dtype=int32)  # the first item is the shape of the Param, the second is its kind
>>> y = tc.max(x); y
y(1, variable, dtype=int32)  # if the Param is 0-dimensional, the first item is its value
>>> y2 = x.max(); y2.item()  # the same as y; use item() to get the value of a Param of size 1
1

"Pass backwards the gradient with regard to a scalar Param."
>>> y.backward(); x.grad  # compute gradients wrt `y` and generates related trainable Params
<generator object ...>
array([0., 0., 1.])
>>> x.del_grad(); x.grad
0
>>> y = tc.reLU(x); y.data  # check the data in the form of an array
array([0, 0, 1])
>>> (x.relu() == y).all().item()  # method names are lowercase
True
>>> y.sum().backward(); x.grad  # note that autograd must start from a scalar
<generator object ...>
array([0., 1., 1.])

"Convert an array to Param. Very easy since Param is a subclass of np.ndarray."
>>> import numpy as np
>>> y = np.random.randint(3, size=3); y
array([2, 1, 2])
>>> y = tc.utils.onehot(y, k=3); y
array([[0, 0, 1],
       [0, 1, 0],
       [0, 0, 1]])
>>> z = Param(y); z
P416(<3, 3>, variable, dtype=int32)
>>> z[0, 0] = 666; y[0, 0]  # data is shared
666
>>> z2 = y.view(Param); z2  # can also use `view()`, this creates a constant Param
z2(<3, 3>, constant, dtype=int32)
>>> z2[0, 1] = 999; z[0, 1]
999

"Fill up a Param."
>>> Param(0, size=[3, 3]).data
array([[0., 0.],
       [0., 0.]])
>>> Param([1, 2, 3], size=[2, 3], dtype=int).data
array([[1, 2, 3],
       [1, 2, 3]])

"Create a random Param."
>>> x = Param(size=[10, 4], kind=0); x  # 0: constant, 1: variable, 2: trainable
x(<10, 4>, constant)
>>> x.shape
(10, 4)
>>> Param(size=[10, 4], mean=1, scale=1, name='x1')  # specify mean, scale and name
x1(<10, 4>, trainable)  # the default kind of a random Param is "trainable"
>>> w = Param(size=[4, 3])
>>> b = Param(size=3)

"Let's have some SGD! 😁"
>>> y = tc.utils.onehot(np.random.randint(3, size=10), k=3)  # OK to directly use numpy arrays
>>> e = (x @ w + b).smce(y); e  # softmax cross-entropy
e(1.03, variable)
>>> tc.utils.graph.show_graph(e)  # see the graph below
<graphviz.dot.Digraph object at ...>
>>> def SGD(pars):
        for p in pars:
            p -= 1e-3 * p.grad
            p.del_grad()
>>> pars = list(e.backward()); pars
[b(<3>, trainable), w(<4, 3>, trainable)]
>>> SGD(pars)
```

![A simple computation graph](compgraph.png)

```python
"More about Param functions."
>>> z.data
array([[666, 999,   1],
       [  0,   1,   0],
       [  0,   0,   1]])
>>> z.argmax(axis=0)  # all numpy array methods are available to Param since it subclasses np.ndarray
array(<3>, constant, dtype=int64)  # the output will be a constant Param
>>> z.reshape(-1)  # some methods (like `sum`, `max` shown above) have been overridden to support autograd
P536(<9>, variable, dtype=int32)  # the output will be a variable Param
>>> z.reshape(-1)._ctx  # the operation (as the "context" of the output) is stored for autograd
reshape(z(<3, 3>, variable, dtype=int32), -1)

"Some functions can be initialized in addition to being applied directly."
>>> np.all(tc.affine(x, w, b) == x @ w + b).item()
True
>>> affine = tc.affine(2)  # initialize the function if the input Param is absent
>>> hasattr(affine, 'w')
False
>>> x.shape, affine(x)  # affine(2) initializes an affine map of output dimensionality = 2
((10, 4), P520(<10, 2>, variable))
>>> hasattr(affine, 'w') and affine.w
P624(<4, 2>, trainable)  # Params in these functions get initialized only after receiving inputs 
>>> np.all(affine(x) == x @ affine.w + affine.b).item()
True
>>> imgs = Param(size=[100, 3, 256, 256], kind='constant')
>>> conv = tc.conv2D(32, size=(5, 5), stride=2)
>>> h = conv(imgs); h  # inform `conv` the input channels and then it initialize filters of the correct shape
h(<100, 32, 126, 126>, variable)
>>> conv.filters
P560(<32, 3, 5, 5>, trainable)
>>> xn = tc.normalize(x)
>>> xn.mean().item(), xn.std().item()
(0., 1.)  # actually has some tiny errors
>>> layerNorm = tc.normalize(axis=0)
normalize()  # now it is a function containing trainable weight and bias
>>> layerNorm(x); x.shape
(10, 4)
>>> layerNorm.w, layerNorm.b  # Params initialized according to x.shape
(P304(<1, 4>, trainable), P416(<1, 4>, trainable))
>>> a = Param(1, size=[100, 10]); a.sum().item()
1000
>>> np.count_nonzero(tc.dropout(a))
474  # dropout rate = 0.5 by default
>>> np.count_nonzero(tc.dropout(a, p=0.3))
713
>>> dropout = tc.dropout(p=0.2)
>>> np.count_nonzero(dropout(a))
802

"Compose several functions to build a neural network!"
>>> from toych.func import *
>>> from toych.optim import Adam
>>> nn = tc.model.Compose(
        conv2D(32, size=(5, 5), stride=2),
        normalize2D(), maxPool(),
        conv2D(64, size=(5, 5), stride=2),
        normalize2D(), maxPool(),
        conv2D(128, size=(3, 3), stride=2),
        normalize2D(), maxPool((4, 4)),
        flatten,
        affine(64), normalize(),
        dropout, reLU,
        affine(32), normalize(),
        dropout(p=0.2), reLU,
        affine(10), softmax
    )
>>> labels = tc.utils.onehot(np.random.randint(10, size=100), 10)
>>> optimizer = Adam()  # no need to pass in Params when init
>>> nn(imgs)  # quite slow!
P215(<100, 10>, variable)
>>> for epoch in range(10):
        preds = nn(imgs)
        loss = preds.crossentropy(labels)
        optimizer(loss.backward())

"Or simply use the `fit` method of a Model."
>>> nn.fit(imgs, labels, epochs=10, optimizer=optimizer, loss='crossentropy')
```

## TODO

* [ ] let toych play with 2048
* [ ] FIX conv2D
* [ ] implement VAE
* [ ] implement GAN
* [ ] implement ResNet
* [ ] implement a transformer

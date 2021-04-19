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

"Create a Param."
>>> Param.auto_name = True  # automatically name the Param (only for fun, sometimes slow)
>>> x = Param([1, 2, 3]); x
x(<3>, variable, dtype=int32)  # the first item is the shape of the Param, the second is its kind
>>> y = tc.max(x); y
y(3, variable, dtype=int32)  # if the Param is a single number, the first item is just its value
>>> y2 = x.max()  # the same as y

"Pass backwards the gradient with regard to a scalar Param."
>>> y.backward()  # returns a generator of trainable Params
<generator object ...>
>>> x.grad  # its gradient with regard to y
array([0., 0., 1.])
>>> x.zero_grad(); x.grad
0
>>> x.sum().backward(); x.grad
array([1., 1., 1.])

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
x1(<10, 4>, trainable)  # by default, the kind is "trainable"
>>> x.data  # check the data in the form of an array
array([...])
>>> w = Param(size=[4, 3])
>>> b = Param(size=3)

"Let's have some SGD! 😁"
>>> y = tc.utils.onehot(np.random.randint(3, size=10), k=3)  # OK to directly use numpy arrays
>>> e = (x @ w + b).smce(y)  # softmax cross-entropy
>>> tc.utils.graph.show_compgraph(e)  # see the graph below
<graphviz.dot.Digraph object at ...>
>>> def SGD(pars):
        for p in pars:
            if p.trainable:
                p -= 1e-3 * p.grad
            p.zero_grad()
>>> SGD(e.backward())

"More Param functions!"
>>> z.data
array([[666, 999,   1],
       [  0,   1,   0],
       [  0,   0,   1]])
>>> z.argmax(axis=0)  # all numpy array methods are available to Param since it subclasses np.ndarray
array(<3>, constant, dtype=int64)  # the outputs will be constant Params
>>> z.reshape(-1)  # some methods (like `sum`, `max` mentioned above) have been overridden to support autograd
P536(<9>, variable, dtype=int32)  # the outputs will be variable Params
>>> z.reshape(-1)._ctx  # the operation will be stored for future backprop of gradients
reshape(z(<3, 3>, variable, dtype=int32), -1)

"Some functions can be initialized to have trainable Params in addition to being applied directly."
>>> np.all(tc.Affine(x, w, b) == x @ w + b)
array(True, constant, dtype=bool)
>>> affine = tc.Affine(2)  # pass the output dimension to init an Affine function containing trainable Params
>>> hasattr(affine, 'w')
False
>>> x.shape, affine(x)
((10, 4), P520(<10, 2>, variable))
>>> hasattr(affine, 'w') and affine.w
P624(<4, 2>, trainable)  # these kinds of toych functions init Params only after receiving inputs 
>>> np.all(affine(x) == x @ affine.w + affine.b).item()
True
>>> imgs = Param(size=[100, 3, 256, 256], kind='constant')
>>> conv = tc.Conv2D(32, size=(5, 5), stride=2)
>>> h = conv(imgs); h  # inform `conv` the input channels and let it init filters
h(<100, 32, 126, 126>, variable)
>>> conv.filters
P560(<32, 3, 5, 5>, trainable)
>>> xn = tc.normalize(x)
>>> xn.mean().item(), xn.std().item()
(0., 1.)  # actually has some tiny errors
>>> Norm = tc.normalize(axis=0)
normalize()  # now it is a function containing trainable weight and bias
>>> Norm(x); x.shape
(10, 4)
>>> Norm.w, Norm.b
(P304(<1, 4>, trainable), P416(<1, 4>, trainable))
>>> a = Param(1, size=[100, 10]); a.sum().item()
1000
>>> np.count_nonzero(tc.dropout(a))
474  # dropout rate = 0.5 by default
>>> np.count_nonzero(tc.dropout(a, p=0.3))
713
>>> dropout = tc.dropout(p=0.2)  # some functions are "partial" - refer to functools.partial for details
>>> np.count_nonzero(dropout(a))
802

"Compose several functions to build a neural network!"
>>> from toych.func import *
>>> nn = tc.model.Compose(
        Conv2D(32, size=(5, 5), stride=2),
        Conv2D(64, size=(5, 5), stride=2),
        Conv2D(128, size=(3, 3), stride=2),
        MaxPool2D((4, 4)),
        flatten,
        Affine(64), normalize(),
        dropout, ReLU,
        Affine(32), normalize(),
        dropout(0.2), ReLU,
        Affine(10), softmax
    )
>>> pred = nn(imgs)  # quite slow!
pred(<100, 10>, variable)
>>> y = tc.utils.onehot(np.random.randint(10, size=100), 10)
>>> loss = pred.crossentropy(y)
>>> SGD(loss.backward())
```

![A simple computation graph](compgraph.png)

## TODO

* [ ] let toych play with 2048
* [ ] FIX Conv2D
* [ ] implement VAE
* [ ] implement GAN
* [ ] implement ResNet
* [ ] implement a transformer

from importer import *

nn = Compose(
    conv2D(32, 3, stride=2),
    maxPool(2), normalize2D(),
    flatten,
    leakyReLU, affine(64),
    leakyReLU, affine(10)
)

ims = Param((100, 3, 32, 32), kind=0)
labels = Param(utils.onehot(np.random.randint(10, size=100), 10))

loss = nn(ims).smce(labels)
loss.backward()
show_compgraph(loss)
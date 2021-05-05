from importer import *
from data.mnist import *


# x_train, x_test = x_train.reshape(-1,1,*im_size), x_test.reshape(-1,1,*im_size)
y_train, y_test = [onehot(y, 10) for y in (y_train, y_test)]

classes = 10

classifier = Compose(
    affine(128),
    normalize, reLU,
    affine(classes)
)

generator = model.VAE(
    Compose(
        affine(128), leakyReLU,
        affine(64), leakyReLU,
    ),  # encoder
    Compose(
        affine(128), leakyReLU(0.1),
        affine(256), leakyReLU(0.1),
        affine(512), leakyReLU(0.1),
        affine(1024), leakyReLU(0.1),
        affine(784), sigmoid
    ),  # decoder
    latent_dim = 2
)

optimizer = optim.Adam(1e-3, reg='l2')

def plot_img(img, ax=plt):
    img = img.reshape(im_size)
    ax.imshow(img)

def show_contrast(**kwds):
    plot_img(eg_img, plt.subplot(1, 2, 1))
    plot_img(generator.eval(eg_img), plt.subplot(1, 2, 2))
    plt.show()

eg_img = x_test[random.randrange(1000)]

with Profile('train generator'):
    generator.fit(x_train, epochs=8, bs=100, optimizer=optimizer,
                  callbacks=[show_contrast], showgraph=0)

# generator.save('mnist-gen')


def accuracy(out, labels):
    return (out.argmax(axis=1) == labels.argmax(axis=1)).astype(float).mean()

(x_train, y_train), (x_val, y_val) = train_val_split(x_train, y_train)

with Profile('train classfier'):
    classifier.fit(x_train, y_train, epochs=3,
                   loss='smce', bs=128, lr=0.5, optimizer='sgd',
                   val_data=(x_val, y_val), metrics={'val_acc': accuracy})

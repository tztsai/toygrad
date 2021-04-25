from importer import *
from data.mnist import *

def classifier(classes):
    return Compose(
        conv2D(32, 3, stride=2, normalize=True),
        reLU, maxPool(2),
        conv2D(64, 3, stride=2, normalize=True),
        reLU, flatten,
        affine(classes)
    )

clf = classifier(10)

x_train, x_test = x_train.reshape(-1,1,*im_size), x_test.reshape(-1,1,*im_size)
y_train, y_test = [onehot(y, 10) for y in (y_train, y_test)]
(x_train, y_train), (x_val, y_val) = train_val_split(x_train, y_train)


def accuracy(out, labels):
    return (np.argmax(out, axis=1) == np.argmax(labels, axis=1)).astype(float).mean()

history = clf.fit(x_train, y_train, epochs=10, loss='smce',
                  val_data=(x_val,y_val), metrics={'val_acc': accuracy})

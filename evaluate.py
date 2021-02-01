import matplotlib.pyplot as plt
from utils import *
import random
from nn import Sequential, Dense

test_binary_clf = False
test_mnist = True


if test_binary_clf:
    data, labels = join_classes(
        normal(50, [0, 2], 0.5),
        normal(50, [-2, 0], 0.5),
        normal(50, [0, -2], 0.5),
        normal(50, [2, 0], 0.5),
        labels=[1, -1, 1, -1]
    )
    # plot_dataset(data, labels)
    # plt.show()

    model = Sequential(2, Dense(6), Dense(6), Dense(1),
                       activation='tanh', lr=5e-3)
    model.fit(data, labels, epochs=100, callbacks=[
              AnimStep(data, labels, binary=True)])


if test_mnist:
    from pathlib import Path
    import requests
    import pickle
    import gzip

    DATA_DIR = Path('./')
    URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
    FILENAME = "mnist.pkl.gz"
    PATH = DATA_DIR / FILENAME

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not PATH.exists():
        content = requests.get(URL + FILENAME).content
        PATH.open("wb").write(content)

    with gzip.open(PATH.as_posix(), "rb") as f:
        ((x_train, y_train), (x_test, y_test), _) = pickle.load(f, encoding="latin-1")

    im_size = (28, 28)
    arr_size = np.prod(im_size)

    model = Sequential(arr_size, Dense(30), Dense(10), activation='tanh')
    model.fit(x_train, onehot(y_train, 10), epochs=30)

    output = model.forward(x_test)
    print('MNIST accuracy:', np.mean(np.argmax(output, axis=1) == y_test))

    # visualize prediction
    x_sample = x_test[random.sample(range(100), 8)]
    output = np.argmax(model.forward(x_sample), axis=1)

    for i in range(8):
        # plot image
        ax = plt.subplot(8, 2, 2 * i + 1)
        ax.axis('off')
        ax.imshow(x_sample[i].reshape(im_size), cmap='gray')

        # plot prediction
        ax = plt.subplot(8, 2, 2 * i + 2)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.axis('off')
        ax.text(0, 0, str(output[i]), fontsize=15)

    plt.show()

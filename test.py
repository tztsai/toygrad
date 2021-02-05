# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from IPython.display import HTML
from utils import *
from nn import Sequential, Dense


# %% Linearly Inseparable Data
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

anim_step = AnimStep(data, labels, binary=True)
model.fit(data, labels, epochs=30, callbacks=[anim_step])


# %% MNIST digit classification
from pathlib import Path
import requests
import pickle
import gzip

DATA_DIR = Path('.')
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

x_sample = x_test[np.random.randint(1000, size=8)]


# %%
model = Sequential(arr_size, Dense(30), Dense(10), activation='tanh')
model.fit(x_train, onehot(y_train, 10), epochs=30)

output = model.forward(x_test)
print('MNIST accuracy:', np.mean(np.argmax(output, axis=1) == y_test))

# visualize prediction
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


# %% Auto Encoder
autoencoder = Sequential(arr_size, 30, 10, 30, arr_size,
                         activation='logistic', lr=5e-3)
autoencoder.fit(x_train, x_train, epochs=50)

output = autoencoder.forward(x_sample)

for i in range(8):
    # plot original image
    ax = plt.subplot(8, 2, 2 * i + 1)
    ax.axis('off')
    ax.imshow(x_sample[i].reshape(im_size), cmap='gray')

    # plot reconstructed image
    ax = plt.subplot(8, 2, 2 * i + 2)
    ax.axis('off')
    ax.imshow(output[i].reshape(im_size), cmap='gray')

plt.show()


# %%
n = 8
xrange = np.linspace(0, 1, num=n)
# sample_ids = np.random.randint(100, size=n)
sample_ids = range(8)
x_sample = autoencoder.forward(x_test[sample_ids])
encoding = autoencoder.layers[1]._output
perturbed_dims = [2, 6]

def decode(encoding):
    return autoencoder.forward(encoding, 2).reshape(-1, *im_size)

def perturb(encoding, d, x):
    e = np.copy(encoding)
    e[:, d] = x
    return e

fig, axs = plt.subplots(n, n)

# for i, xi in enumerate(xrange):
#     for j, xj in enumerate(xrange):
#         e = perturb(encoding, perturbed_dims[0], xi)
#         e = perturb(e, perturbed_dims[1], xj)
#         axs[i][j].axis('off')
#         axs[i][j].imshow(decode(e), cmap='gray')

for i, xi in enumerate(xrange):
    e = perturb(encoding, perturbed_dims[0], xi)
    out = decode(e)
    for j in range(n):
        axs[i][j].axis('off')
        axs[i][j].imshow(out[j], cmap='gray')

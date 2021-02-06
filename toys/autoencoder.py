#%%
import sys, os
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
import requests
from pathlib import Path
from models import Sequential


#%% Download dataset
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
input_dim = np.prod(im_size)


# %% Auto Encoder
autoencoder = Sequential(input_dim, 30, 10, 30, input_dim, activation='logistic')
autoencoder.fit(x_train, x_train, epochs=10)

x_sample = x_test[np.random.randint(1000, size=8)]
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

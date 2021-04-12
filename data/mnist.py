import numpy as np
from pathlib import Path
import requests, os
import pickle
import gzip

DATA_DIR = Path('.') / 'data'
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

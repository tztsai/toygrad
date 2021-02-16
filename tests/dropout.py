import init
from models import Sequential
from layers import Dense
from utils import onehot
import numpy as np
from tensorflow.keras.datasets import mnist

(x_tr, y_tr), (x_ts, y_ts) = mnist.load_data()
im_shape = x_tr[0].shape
im_size = np.prod(im_shape)

def accuracy(model):
    return np.mean(np.argmax((model(x_ts)), axis=(-1)) == y_ts)

nn = Sequential(im_size, Dense(30, activation='tanh'), Dense(10, activation='logistic'))
x_tr = x_tr.reshape(-1, im_size)
x_ts = x_ts.reshape(-1, im_size)
nn.fit(x_tr, (onehot(y_tr, 10)), epochs=10)

print(accuracy(nn))
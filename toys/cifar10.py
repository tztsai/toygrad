#%%
import pickle
import random
import numpy as np
from functools import partial
from importer import *
import gc

gc.set_debug(1)

#%%
DATAPATH = 'data/cifar10/'

def load_data_batch(filename):
    with open(DATAPATH + filename, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return tuple(np.asarray(data[k]) for k in
                 [b'data', b'labels'])

def train_val_split(inputs, labels, ratio=0.8):
    N = len(inputs)
    idx = random.sample(range(N), int(ratio*N))
    itr = np.zeros(N, dtype=bool)
    itr[idx] = 1
    return (inputs[itr], labels[itr]), (inputs[~itr], labels[~itr])

def perprocess(*datasets):
    return [ims.reshape(-1, 3, 32, 32) for ims in standardize(*datasets)]
    
#%%
xs, ts = zip(*[load_data_batch(f'data_batch_{i+1}') for i in range(3)])
x_tr, t_tr = np.vstack(xs), np.hstack(ts)
(x_tr, t_tr), (x_va, t_va) = train_val_split(x_tr, t_tr)
x_ts, t_ts = load_data_batch('test_batch')

x_tr, x_va, x_ts = perprocess(x_tr, x_va, x_ts)
t_tr, t_va, t_ts = map(partial(onehot, k=len(set(t_tr))), [t_tr, t_va, t_ts])

#%%
# nn = ResNet(18)
nn = Compose(
    conv2D(32, 3, stride=2, normalize=True),
    reLU, maxPool(2),
    conv2D(64, 3, stride=2, normalize=True),
    reLU, maxPool(2),
    flatten, affine(10)
)

nn.fit(x_tr, t_tr, epochs=10, lr=8e-3, loss='smce')

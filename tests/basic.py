#%%
import init

from models import Sequential, Dense
from utils import *

xtr, ytr = np.loadtxt('xt'), np.loadtxt('yt')

nn = Sequential(2, 8, 1, activation='tanh')

#%%
nn.fit(xtr, ytr, lr=5e-3, epochs=50, callbacks=[train_anim(xtr, ytr)])
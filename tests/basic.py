#%%
import init

from models import Sequential, Dense
from utils import *

xtr, ytr = np.loadtxt('data/xt'), np.loadtxt('data/yt')
nn = Sequential(2, Dense(8, dropout=0.01), 1, activation='tanh')

#%%
nn.fit(xtr, ytr, lr=5e-3, epochs=50, callbacks=[pred_anim(xtr, ytr)])
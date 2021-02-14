import init

from models import Sequential, Dense
from utils import *

n = 100
muA = np.array([0.5, 1])
sigmaA = 0.3
muB = np.array([-1.0, 0.0])
sigmaB = 0.3

classA = gaussian(n, muA, sigmaA)
classB = gaussian(n, muB, sigmaB)
classA2 = gaussian(n, -2 * muA, sigmaA)

xtr, ytr = join_classes(classA, classB, classA2, labels=[1, -1, 1])

nn = Sequential(2, 15, 1, activation='tanh')
nn.fit(xtr, ytr, lr=2e-3, epochs=100, callbacks=[train_anim(xtr, ytr)])
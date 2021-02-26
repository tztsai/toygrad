# %%
import init
from node import *
from model import *
from activation import *

# %%
a = Linear(2, 5)
tanh1 = Tanh()
b = Linear(2)
tanh2 = Tanh()
c = Linear(5)
sm = SoftMax(5)

a.connect(tanh1).connect(b).connect(tanh2).connect(c).connect(sm)
model = Model(a, sm)

# %%
print(model(rand.rand(10, 5)))
from importer import *

a = Param(np.random.rand(10, 5))
b = Param((5, 3))
c = (a @ b).sum()
pars = list(c.backward())
assert len(pars) == 1 and id(pars[0]) == id(b)

assert a.training
with Param.not_training():
    assert not a.training
assert a.training

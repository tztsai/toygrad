from importer import *

utils.dev.time.sleep(1)

class F(Operation):
    need_init = 1
    def apply(self, x, k=1):
        return k * x
    
f = F(k=2)

assert mean(np.array([1,2,3])) == 2.
assert f(Param(100)) == 200
assert F(Param(100), 3) == 300

class G(Parametrized, Operation):
    def __init__(self):
        self.w = Param([1, 2, 3])
    def update_args(self, x):
        return super().update_args(x, w=self.w)
    def apply(self, x, w):
        assert all(not isinstance(a, Param) for a in [x, w])
        return x + w

g = G()
assert (g(Param([3])) == G(Param([3]), g.w)).all().item()
    
class H(Parametrized, Function):
    def __init__(self):
        self.w = Param([1, 2, 3], dtype=float)
    def update_args(self, x):
        return super().update_args(x, w=self.w)
    def apply(self, x, w):
        return x + w

h = H()
assert (h(3) == [4, 5, 6]).all().item()
save(Param((3, 4)))
save(h, 'fn_h')
h = load('fn_h')
assert h(3).sum().item() == 15

assert affine(np.zeros([5, 3]), Param((3, 4)), Param(size=4)).shape == (5, 4)

aff = affine(10)
assert aff(np.zeros([5, 3])).shape == (5, 10)

a = meanPool(4, stride=2)
p = a(Param(size=[1,3,10,10]))
assert p.shape == (1, 3, 1, 1)

a, b, c = map(Param, [(10, 5), (10, 5), (10, 5)])
max([a, b, c]).backward()
assert {a.grad.max(), b.grad.max(), c.grad.max()} == {0, 1}
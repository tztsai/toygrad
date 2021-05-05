from importer import *

utils.dev.time.sleep(1)

class F(Operation):
    need_init = 1
    def apply(self, x, a, k=1):
        return k*(x + a)
    
f = F(3, k=1)

assert mean(np.array([1,2,3])) == 2.
assert f(Param(100)) == 103.
assert F(Param(100), 13) == 113.

class G(Operation):
    def __init__(self):
        self.w = Param([1, 2, 3])
    def update_args(self, x):
        return super().update_args(x, w=self.w)
    def apply(self, x, w):
        assert all(not isinstance(a, Param) for a in [x, w])
        return x + w

g = G()
repr(g); repr(g)
assert (g(Param([3])) == G(Param([3]), g.parent.w)).all().item()
    
class H(Function):
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
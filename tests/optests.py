from toych import *
setloglevel('INFO')

class F(Operation):
    partial = 1
    def apply(self, x, a, k=1):
        return k*(x + a)
    
assert mean(np.array([1,2,3])) == 2.
f = F(3, k=1)
f2 = F(3)
f3 = f2(k=2)
assert f(Param(100)) == 103.
assert F(Param(100), 13) == 113.
assert f3(Param(100)) == 206.

class G(Operation):
    def __init__(self, n):
        self.w = Param(size=n)
        super().__init__(self.w)
    # def update_args(self, *args, **kwds):
    def apply(self, x, w):
        return x + w
    
g = G(10)
print(g(111))
    
class F2(Function):
    def __init__(self, n):
        # super().__init__()
        self.w = Param(size=n)
    def apply(self, x):
        return x + self.w
    
f2 = F2(10)
print(f2(111))

print(Affine(np.zeros([5, 3]), Param(size=[3, 4]), Param(size=4)).shape)
aff = Affine(10)
print(aff(np.zeros([5, 3])).shape)

a = MeanPool2D((3, 3))
b = a(stride=2)
c = b(Param(size=[1,3,10,10]))
print(c.shape)
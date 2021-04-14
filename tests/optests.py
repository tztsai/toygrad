from init import *


dev.setloglevel('INFO')

class F(Operation):
    partial = 1
    def apply(self, x, a, k=1):
        return k*(x + a)
    
assert mean(np.array([1,2,3])) == 2.
f = F(3, k=1)
print(f(100))
print(F(Param(size=2), 13))
p = Param(size=2)
with Param.not_training():
    p += 1

class G(Operation):
    def __init__(self, n):
        self.w = Param(size=n)
    def apply(self, x):
        return x + self.w
    
g = G(10)
print(g(111))
    
class F2(Function):
    def __init__(self, n):
        self.w = Param(size=n)
    def apply(self, x):
        return x + self.w
    
f2 = F2(10)
print(f2(111))

print(Affine(np.zeros([5, 3]), Param(size=[3, 4]), Param(size=4)).shape)
aff = Affine(10)
print(aff(np.zeros([5, 3])).shape)
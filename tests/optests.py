from init import *

assert mean(np.array([1,2,3])) == 2.

dev.setloglevel('INFO')

class F(Operation):
    partial = 1
    def apply(self, x, a, k=1):
        return k*(x + a)
    
f = F(3, k=1)
print(f(100))
print(F(Param(size=2), 13))

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

class H(Function):
    def apply(self, x, y):
        return x + y

print(H(1, 2))
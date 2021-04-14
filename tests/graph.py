from init import *

A1 = Affine(10)
A2 = Affine(10)
B = Affine(4)
x = Param(size=[10, 3])
h = Param(size=[10, 10])

for _ in range(5):
    h = A1(x) + A2(h)
    
e = B(h).mean()
graph.show_compgraph(e)
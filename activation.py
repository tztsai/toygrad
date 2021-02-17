# from node import Node
import numpy as np


class Activation:
    def __new__(cls, obj):
        if type(obj) is str:
            s = obj.lower()
            if s == 'tanh':
                return Tanh()
            elif s in ['logistic', 'sigmoid']:
                return Logistic()
            elif s == 'relu':
                return ReLU()
            elif s == 'linear':
                return None
            elif s == 'default':
                return False
            else:
                raise ValueError(f"unknown activation function: {obj}")
        elif not obj:
            return None
        else:
            raise ValueError(f"unknown activation function: {obj}")


if __name__ == '__main__':
    sigma = Node('tanh')
    sigma(2)

    relu = ReLU()
    relu(np.random.rand(10, 3) - 0.5)
    print(relu.backward(np.random.rand(10, 3)))

    sm = SoftMax()
    sm(np.random.rand(10, 3))
    error = np.random.rand(10, 3)
    for y, e, be in zip(sm.output, error, sm.backward(error)):
        d1 = np.diag(y) - np.outer(y, y)
        assert (d1 @ e == be).all()

from node import Node
import numpy as np


class ActivationMeta(type):
    def __call__(self, *args, **kwds):
        if self is not Optimizer:
            opt = object.__new__(self)
            opt.__init__(*args, **kwds) 
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
            raise TypeError(f"Activation() argument 1 must be str or Activation")


class Tanh(Node):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, error):
        return error * (1 - self.output**2)


class Logistic(Node):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, error):
        return error * self.output * (1 - self.output)


class ReLU(Node):
    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, error):
        return error * (self.output > 0)


class SoftMax(Node):
    def forward(self, x):
        ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return ex / np.sum(ex, axis=-1, keepdims=True)

    def backward(self, error):
        # TODO: not correct?
        dp = np.sum(error * self.output, axis=-1, keepdims=True)
        return (error - dp) * self.output


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

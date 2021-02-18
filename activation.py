from node import Node
import numpy as np
from utils import make_meta


def get_activation(name: str):
    if name == 'tanh':
        return Tanh
    elif name in ['logistic', 'sigmoid']:
        return Logistic
    elif name == 'relu':
        return ReLU
    elif name == 'linear':
        return None
    elif name == 'default':
        return False
    else:
        raise ValueError(f"unknown activation function: {name}")


class Activation(Node, metaclass=make_meta(get_activation)):
    def setup(self):
        super().setup()
        self.dim_out = self.dim_in


class Tanh(Activation):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, error):
        return error * (1 - self.output**2)


class Logistic(Activation):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, error):
        return error * self.output * (1 - self.output)


class ReLU(Activation):
    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, error):
        return error * (self.output > 0)


class SoftMax(Activation):
    def forward(self, x):
        ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return ex / np.sum(ex, axis=-1, keepdims=True)

    def backward(self, error):
        # TODO: not correct?
        dp = np.sum(error * self.output, axis=-1, keepdims=True)
        return (error - dp) * self.output


if __name__ == '__main__':
    sigma = Activation('tanh')
    print(sigma(2))

    relu = ReLU()
    relu(np.random.rand(10, 3) - 0.5)
    print(relu.backward(np.random.rand(10, 3)))

    sm = SoftMax()
    sm(np.random.rand(10, 3))
    error = np.random.rand(10, 3)
    for y, e, be in zip(sm.output, error, sm.backward(error)):
        d1 = np.diag(y) - np.outer(y, y)
        assert (d1 @ e == be).all()

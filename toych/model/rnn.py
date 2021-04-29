from .basic import *


class LSTM(Model):
    """Long-short term memory."""

    def __init__(self, dim):
        self.w = [affine(dim) for i in range(4)]
        self.u = [affine(dim) for i in range(4)]
        self.h, self.c = Param(0, size=[1, dim]), Param(0, size=[1, dim])

    def apply(self, x):
        wf, wi, wo, wc = self.w
        uf, ui, uo, uc = self.u
        h = self.h
        f = sigmoid(wf(x) + uf(h))  # forget
        i = sigmoid(wi(x) + ui(h))  # input
        o = sigmoid(wo(x) + uo(h))  # output
        c = tanh(wc(x) + uc(h))     # candidate
        self.c = self.c*f + i*c     # update
        self.h = o * tanh(self.c)
        # setparnames()
        return o

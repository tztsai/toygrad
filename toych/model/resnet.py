from .basic import *


class ResNet(Compose):
    config = {
        18: ((64, 2), (128, 2), (256, 2), (512, 2)),
        34: ((64, 3), (128, 4), (256, 6), (512, 3))
    }

    class Block(Compose):
        def __init__(self, c_in, c_out, size):
            super().__init__(
                conv2D(c_out, size), normalize2D(),
                reLU,
                conv2D(c_out, size), normalize2D(),
            )
            if c_in != c_out:  # convolve with filters of size 1 to change number of channels
                self.identity = conv2D(c_out, 1, normalize=True)

        @staticmethod
        def identity(x): return x

        def apply(self, input):
            return (self(input) + self.identity(input)).relu()

    def __init__(self, layers):
        if layers in self.config:
            config = self.config[layers]
        else:
            raise ValueError('ResNet of %d layers is not available' % layers)

        c_outs = [c_out for c_out, n_blocks in config for _ in range(n_blocks)]
        c_in_c_outs = zip([None] + c_outs, c_outs)

        head = Compose(conv2D(64, 7, stride=2, normalize=True),
                       reLU, maxPool(size=(3, 3)))
        body = Compose(*[ResNet.Block(c_in, c_out, 3)
                         for c_in, c_out in c_in_c_outs])
        tail = Compose(meanPool(size=(2, 2)), affine(10))

        super().__init__(head, body, tail)


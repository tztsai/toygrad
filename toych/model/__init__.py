from ..core import AbstractFunction
from ..func import *
from ..optim import *
from ..utils.dev import defaultdict, info, dbg, warn, pbar
from ..utils import BatchLoader, setparnames, graph, pickle


class Model(AbstractFunction):
    """ Baseclass of learning models.
    Wrap any function `f` by `Model(f)` to convert it to a model.
    """
    def __init__(self, func):
        self.apply = func
        
    def apply(self, *args, **kwds):
        raise NotImplementedError
    
    def fit(self, input, target, *, epochs=20, lr=None, bs=None, optimizer='adam',
            loss='l2', val_data=None, metrics={}, callbacks=()) -> dict:
        """ Given the input data, train the parameters to fit the target data.

            Args:
                input: an array of input data
                target: an array of target or label data
                epochs: number of epochs to train
                lr: learning rate, use lr of the optimizer by default
                bs: batch size, use bs of BatchLoader by default
                optimizer (Optimizer): optimizer of the parameters
                loss: the metric to measure the training loss (does not affect backprop!)
                val_data: validation data in the form of (x_val, t_val)
                metrics (dict of {name:function}): functions to be applied to (y_val, t_val),
                    whose outputs will be tracked in the training history
                callbacks (list of function): functions to be called at the end of each epoch,
                    each function taking the NN object as input

            Returns:
                A dict of training history including losses etc.
        """
        input, target = np.asarray(input), np.asarray(target)
        assert input.shape and target.shape

        batches = BatchLoader(input, target, batch_size=bs)
        optimizer = self.getoptim(optimizer, lr=lr)
        loss_func = self.getloss(loss)
        history = defaultdict(list)

        info('\nStart training %s', self)
        info('Input shape: %s', input.shape)
        info('Target shape: %s', target.shape)
        info('Total epochs: %d', epochs)
        info('Batch size: %d', batches.batch_size)
        info('Optimizer: %s', optimizer)

        for epoch in range(epochs):
            info('\nEpoch %d:', epoch)
            
            loss = 0
            for x, t in pbar(batches):
                y = self(x)             # pass forward the input
                e = loss_func(y, t)     # compute the loss
                params = e.backward()   # pass backward the loss
                optimizer(params)
                loss += e

            history['loss'].append(loss / batches.size)

            with Param.not_training():
                if val_data:
                    x_val, t_val = val_data
                    with Param.not_training():
                        y_val = self(x_val)
                    metrics['val_loss'] = loss_func
                    for name, metric in metrics.items():
                        history[name].append(metric(y_val, t_val))
                for callback in callbacks:
                    callback(**locals())

            info('\t' + ', '.join('%s = %.2f' % (k, v[-1])
                                  for k, v in history.items() if v))

        return dict(history)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, filename_or_bytes):
        if type(filename_or_bytes) is bytes:
            return pickle.loads(filename_or_bytes)
        with open(filename_or_bytes, 'rb') as f:
            return pickle.load(f, encoding='bytes')
        
    def state(self):
        return pickle.dumps(self)

    @staticmethod
    def getloss(obj):
        if callable(obj):
            return obj
        elif type(obj) is str:
            name = obj.lower()
            if name in ['mse', 'l2']:
                return mse
            elif name in ['crossentropy', 'cross_entropy', 'ce']:
                return crossentropy
            elif name in ['softmax_crossentropy', 'softmax_cross_entropy', 'smce']:
                return softmaxCrossentropy
            else:
                raise ValueError(f"unknown loss function: {name}")
        raise TypeError
        
    @staticmethod
    def getoptim(obj, **kwds):
        if callable(obj):
            return obj
        elif type(obj) is str:
            name = obj.lower()
            for k, v in list(kwds.items()):  # remove default kwargs
                if v is None: kwds.pop(k)
            if name == 'sgd':
                return SGD(**kwds)
            elif name == 'adam':
                return Adam(**kwds)
            else:
                raise ValueError(f'unknown optimizer: {name}')
        raise TypeError


class Compose(Model):
    def __init__(self, *functions):
        self.fns = functions
        
    def __getitem__(self, i):
        return self.fns[i]
    
    def apply(self, input):
        for f in self.fns:
            input = output = f(input)
        return output


class ResNet(Compose):
    config = {
        18: ((64, 2), (128, 2), (256, 2), (512, 2)),
        34: ((64, 3), (128, 4), (256, 6), (512, 3))
    }
    
    class Block(Compose):
        def __init__(self, c_in, c_out, size):
            super().__init__(
                conv2D(c_out, size, normalize=True),
                reLU,
                conv2D(c_out, size, normalize=True),
            )
            if c_in == c_out:
                self.res = lambda x: x
            else:
                self.res = conv2D(c_out, 1, normalize=True)
            
        def apply(self, input):
            return (self(input) + self.res(input)).relu()
            
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


class LSTM(Model):
    """Long-short term memory."""

    def __init__(self, h, d):
        self.w = [affine(h) for i in range(4)]
        self.u = [affine(h) for i in range(4)]
        self.h, self.c = Param(0, size=[1, h]), Param(0, size=[1, h])
        
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
        setparnames()
        return o

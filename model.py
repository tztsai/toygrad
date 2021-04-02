from core import *
from utils.dev import dbg, info, pbar
from utils import BatchLoader


class Model:
    def __init__(self, function):
        self.apply = function

    def apply(self, *args, **kwds):
        raise NotImplementedError

    def __call__(self, *args, **kwds):
        return self.apply(*args, **kwds)

    def fit(self, input, target, *, epochs=20, lr=None, bs=None,
            optimizer='sgd', loss='l2', val_data=None, callbacks=()) -> dict:
        """Given the input data, train the parameters to fit the target data.

            Args:
                input: an array of input data
                target: an array of target or label data
                epochs: number of epochs to train
                lr: learning rate, use lr of the optimizer by default
                bs: batch size, use bs of BatchLoader by default
                optimizer (Optimizer): optimizer of the parameters
                loss: the metric to measure the training loss (does not affect backprop!)
                val_data: validation data in the form of (x_val, t_val)
                callbacks (list of function): functions to be called at the end of each epoch,
                    each function taking the NN object as input

            Returns:
                A dict of training history including losses etc.
        """
        input, target = np.asarray(input), np.asarray(target)
        assert input.shape and target.shape

        batches = BatchLoader(input, target, batch_size=bs)
        optimizer = Optimizer(optimizer, lr)
        loss_func = Loss(loss)
        history = {'loss': [], 'val_loss': []}

        info('\nStart training', self)
        info('Input shape:', input.shape)
        info('Target shape:', target.shape)
        info('Total epochs:', epochs)
        info('Batch size:', batches.batch_size)
        info('Optimizer:', optimizer)

        for epoch in range(epochs):
            info('\nEpoch:', epoch)

            loss = 0
            for x, t in pbar(batches):
                y = self(x)             # pass forward the input
                l = loss_func(y, t)     # compute the loss
                l.backward()            # pass backward the loss
                optimizer.update(self.parameters)
                loss += l

            history['loss'].append(loss / len(target))

            if val_data:
                x_val, t_val = val_data
                history['val_loss'].append(self.loss(x_val, t_val))

            info('\t' + ', '.join('%s = %.2f' % (k, v[-1])
                                for k, v in history.items() if v))

            for callback in callbacks:
                callback(locals())

        return history
    
def call_op(self, *args, **kwds):
    if any(isinstance(arg, Model) for arg in args): return Model(self)
    return call_op.orig(self, *args, **kwds)

call_op.orig = Operation.AbstractOp.__call__
Operation.AbstractOp.__call__ = call_op


class Affine(Model):
    def __init__(self, dim_in, dim_out, with_bias=True):
        self.with_bias = with_bias
        self.weights = Parameter(size=[dim_in + with_bias, dim_out])
    
    def apply(self, input):
        if self.with_bias:
            bias, weights = self.weights[0], self.weights[1:]
            return input @ weights + bias
        else:
            return input @ self.weights

class Compose(Model):
    def __init__(self, *funcs):
        self.funcs = funcs

    def apply(self, input):
        for f in self.funcs:
            input = output = f(input)
        return output
    

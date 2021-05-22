from ..core import Function
from ..func import *
from ..optim import *
from ..utils import BatchLoader, graph
from ..utils.dev import defaultdict, info, dbg, warn, progbar, Profile
from ..utils.graph import show_graph


class Model(Function):
    """ Baseclass of learning models.
    Wrap any function `f` by `Model(f)` to convert it to a model.
    """
    blackbox = False
    default_loss = 'mse'
    default_optimizer = 'Adam'
    
    def eval(self, input):
        with Param.not_training():
            return self(input)
    
    def fit(self, input, target, *, epochs=10, lr=None, bs=None, optimizer=None, loss=None, 
            val_data=None, val_bs=500, metrics={}, callbacks=(), showgraph=False) -> dict:
        """ Given the input data, train the parameters to fit the target data.

        Args:
        - input: an array of input data
        - target: an array of target or label data
        - epochs: number of epochs to train
        - lr: learning rate, use lr of the optimizer by default
        - bs: batch size, use bs of BatchLoader by default
        - optimizer (Optimizer): optimizer of the parameters
        - loss: the metric to measure the training loss (does not affect backprop!)
        - val_data: validation data in the form of (x_val, t_val)
        - metrics (dict of {name:function}): functions to be applied to (y_val, t_val),
            whose outputs will be tracked in the training history
        - callbacks (list of function): functions to be called at the end of each epoch,
            each function taking the NN object as input

        Returns: A dict of training history including losses etc.
        """
        input, target = np.asarray(input), np.asarray(target)
        assert input.shape and target.shape

        batches = BatchLoader(input, target, bs=bs)
        optimizer = self.getoptim(optimizer or self.default_optimizer, lr=lr)
        loss_fn = self.getloss(loss or self.default_loss)
        history = defaultdict(list)

        if val_data:
            val_batches = BatchLoader(*val_data, bs=val_bs)
            metrics['val_loss'] = loss_fn

        info('\nTraining model: %s', self)
        info('Input shape:\t%s', input.shape)
        info('Target shape:\t%s', target.shape)
        info('Total epochs:\t%d', epochs)
        info('Batch size:\t%d', batches.bs)
        info('Optimizer:\t%s', optimizer)

        for epoch in range(epochs):
            info('\nEpoch %d:', epoch)
            
            loss = 0
            for x, y in progbar(batches):
                o = self(x)              # pass forward the input
                ls = loss_fn(o, y)       # compute the loss
                params = ls.backward()   # pass backward the loss
                optimizer(params)
                loss += ls.item()
                
                if showgraph:
                    show_graph(ls)
                    showgraph = False

            history['loss'].append(loss / len(batches))

            with Param.not_training():
                if val_data:
                    for name, metric in metrics.items():
                        pairs = [(self(x), y) for x, y in val_batches]
                        score = np.mean([metric(*p) for p in pairs])
                        history[name].append(score)
                for callback in callbacks:
                    callback(**locals())

            info('\t' + ', '.join('%s = %.3e' % (k, v[-1])
                                  for k, v in history.items() if v))

        return dict(history)

    @staticmethod
    def getloss(obj):  # I can override this to add regularization loss
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

import numpy as np
import pickle
from optimizer import Optimizer
from function import Function
from loss import Loss
from node import Node
from utils import *


class Model(Function):
    """A neural network model."""
    
    def __init__(self, entry: Node, exit: Node = None):
        self.entry = entry
        self.exit = entry if exit is None else exit
        
        assert isinstance(entry, Node) and isinstance(exit, Node), \
            'the entry and the exit must be Node instances'
        
        self.nodes = self.traverse()
        self._training = None
        self.training = True
        
        if self.exit not in self.nodes:
            raise AssertionError('the exit is not connected to the entry')
        
        self.entry.setup()
    
    @property
    def training(self):
        return self._training
    
    @training.setter
    def training(self, value):
        if value not in [True, False]:
            raise TypeError('the training attribute should be set to boolean values')
        
        self._training = value
        for node in self.nodes:
            node.training = value
            
    def traverse(self):
        """Search all connected nodes from the entry node."""
        nodes = []
        node_set = set()
        
        def dfs(n: Node):
            if n not in node_set:
                nodes.append(n)
                node_set.add(n)
                for d in n.descendants: dfs(d)
                
        dfs(self.entry)
        return nodes
    
    @property
    def parameters(self):
        for node in self.nodes:
            yield from node.parameters.values()
            
    def forward(self, input):
        """Receive the input and compute the output."""
        self.entry.forward(input)
        return self.exit.output

    def backward(self, error):
        """If in the training mode, propagates back the error and computes parameters' gradients."""
        if not self.training:
            warn('not in training mode, back-prop has no effect')
        else:
            self.exit.backward(error)

    def predict(self, input):
        """Predict the output in the evaluation mode."""
        training = self.training
        self.training = False
        output = self(input)
        self.training = training
        return output

    def loss(self, input, target, loss: Union[Loss, str] = 'l2'):
        """Compute the average loss given the input and the target data."""
        return Loss(loss)(self.predict(input), target) / len(input)

    def state(self):
        return self.__dict__.copy()

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.state(), f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        for attr in state:
            setattr(self, attr, state[attr])


def train(model: Model, input, target, *, epochs=20, lr=None, bs=None,
          optimizer: Union[str, Optimizer] = 'sgd', loss: Union[str, Loss] = 'l2',
          val_data: Optional[list] = None, callbacks: list = ()) -> dict:
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
    
    for arr in [input, target]:
        if dim(arr) == 0:
            raise TypeError('data should be at least 1 dimensional')
        
    # insert a dimension if data is 1D
    input = input[-1, None] if dim(input) == 1 else input
    target = target[-1, None] if dim(target) == 1 else target
    
    batches = BatchLoader(input, target, batch_size=bs)
    optimizer = Optimizer(optimizer, lr)
    loss_func = Loss(loss)
    history = {'loss': [], 'val_loss': []}

    info('\nStart training', model)
    info('Input shape:', input.shape)
    info('Target shape:', target.shape)
    info('Total epochs:', epochs)
    info('Batch size:', batches.batch_size)
    info('Optimizer:', optimizer)

    for epoch in range(epochs):
        info('\nEpoch:', epoch)

        loss = 0
        for xb, tb in pbar(batches):
            yb = model.forward(xb)               # forward pass the input
            # accumulate the loss of the output
            loss += loss_func(yb, tb)
            eb = loss_func.backward()           # the output layer
            model.backward(eb)                   # backprop the error
            optimizer.update(model.parameters)   # update parameters

        history['loss'].append(loss / len(target))

        if val_data:
            x_val, t_val = val_data
            history['val_loss'].append(model.loss(x_val, t_val))

        info('\t' + ', '.join('%s = %.2f' % (k, v[-1])
                                for k, v in history.items() if v))

        for callback in callbacks:
            callback(locals())

    return history

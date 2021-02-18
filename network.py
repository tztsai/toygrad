import numpy as np
import pickle
from optimizers import Optimizer
from function import Function
from loss import Loss
from node import Node
from utils import *


class Network(Function):
    """Base class of a neural network.
    
    Two modes:
        training: gradients are recorded and parameters can be updated
        evaluation: the network only computes the output
    """

    def __init__(self, entry: Optional[Node] = None):
        self.nodes = {}
        self.entry = entry
        self.exit = None
        self.training = True
        if entry: self.add(entry)
        
    def add(self, node: Node):
        if id(node) not in self.nodes:
            self.nodes[id(node)] = node
            node.network = self
            for desc in node.descendants:
                self.add(desc)
        
    @property
    def parameters(self):
        for node in self.nodes.values():
            yield from node.parameters.values()
            
    def forward(self, input):
        """Receive the input and compute the output."""
        self.entry.forward(input)
        if self.exit:
            return self.exit.output

    def backward(self, error):
        """If in the training mode, propagates back the error and computes parameters' gradients."""
        if self.exit:
            self.exit.backward(error)

    def predict(self, input):
        """Predict the output in the evaluation mode."""
        training = self.training
        self.training = False
        output = self.forward(input)
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


def train(model: Network, input, target, *, epochs=20, lr=None, bs=None,
          optimizer: Union[str, Optimizer] = 'sgd', loss: Union[str, Loss] = 'l2',
          val_data: Optional[list] = None, callbacks: list = ()) -> dict:
    """Given the input data, train the parameters to fit the target data.

        Args:
            input: an array of input data - if 1D, then each point is a number;
                if 2D, then each point is a row vector in the array
            target: an array of target or label data - if 1D, then each point is a number;
                if 2D, then each point is a row vector in the array
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
    input, target = reshape2D(input), reshape2D(target)

    batches = BatchLoader(input, target, batch_size=bs)
    optimizer = Optimizer(optimizer, lr)
    loss_func = Loss(loss)
    history = {'loss': [], 'val_loss': []}

    print('\nStart training', model)
    print('Input shape:', input.shape)
    print('Target shape:', target.shape)
    print('Total epochs:', epochs)
    print('Batch size:', batches.batch_size)
    print('Optimizer:', optimizer)

    for epoch in range(epochs):
        print('\nEpoch:', epoch)

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

        print('\t' + ', '.join('%s = %.2f' % (k, v[-1])
                                for k, v in history.items() if v))

        for callback in callbacks:
            callback(locals())

    return history

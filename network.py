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

    def __init__(self, entry=None):
        self.entry = entry
        self.exit = None
        self.nodes = set()
        self.parameters = set()
        self.training = True
        
        if entry: self.add(entry)

    def forward(self, input):
        """Receive the input and compute the output."""
        raise NotImplementedError

    def backward(self, error):
        """If in the training mode, propagates back the error and computes parameters' gradients."""

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
    
    def add(self, node):
        self.parameters.extend(node.parameters.values())

    def state_dict(self):
        return self.__dict__.copy()

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.state_dict(), f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        for attr in state:
            setattr(self, attr, state[attr])


class Sequential(NN):
    """Sequential neural network."""

    @property
    def depth(self):
        return len(self.layers)

    def add(self, layer: Layer):
        if layer._built:
            assert layer.input_dim == self.shape[-1], \
                "cannot match the input dimensionality of %s" % layer
        else:
            layer.setup(self.shape[-1])

        layer.model = self
        self.layers.append(layer)
        self.shape.append(layer.size)
        self.parameters.extend(layer.parameters.values())

        if layer.activation is False:
            layer.activation = self.activation

    def forward(self, input, start_layer=0):
        for k in range(start_layer, self.depth):
            output = self.layers[k].forward(input)
            input = output
        return output

    def backward(self, error):
        for k in reversed(range(self.depth)):
            error = self.layers[k].backward(error, pass_error=bool(k))
            if error is None: break

    def step(self, input, target):
        output = self.forward(input)
        self.backward(output, target)

    def __repr__(self):
        layers_repr = ', '.join(map(repr, self.layers))
        return f'Sequential({self.shape[0]}, {layers_repr})'


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

import numpy as np
from optimizers import Optimizer, SGD
from activations import Activation
from layers import Layer, Dense
from utils import *


class NN(baseclass):
    """Base class of a neural network."""
    
    def __init__(self):
        self.parameters = []
    
    def fit(self, input, target, epochs=20, batch_size=32, optimizer: Optimizer = SGD(),
            val_data=None, callbacks=()) -> dict:
        """Given the input data, train the parameters to fit the target data.
        
        Args:
            input: an array of input data - if 1D, then each point is a number;
                if 2D, then each point is a row vector in the array
            target: an array of target or label data - if 1D, then each point is a number;
                if 2D, then each point is a row vector in the array
            epochs: number of epochs to train  
            batch_size: batch size  
            optimizer (Optimizer): optimizer of the parameters  
            val_data (optional): validation data in the form of (x_val, t_val)  
            callbacks (list of function): functions to be called at the end of each epoch,
                each function taking the NN object as input
                
        Returns:
            A dict of training history including loss etc.
        """
        input = assure_2D(input)
        target = assure_2D(target)

        batches = BatchLoader(input, target, batch_size=batch_size)
        history = {'loss': [], 'val_loss': []}

        for epoch in range(epochs):
            print('\nEpoch:', epoch + 1)

            loss = 0
            for xb, tb in pbar(batches):
                yb = self.forward(xb)
                self.backward(yb, tb)
                optimizer.update(self.parameters)
                loss += self.loss(yb, tb, average=False)

            history['loss'].append(loss / len(target))
            
            if val_data:
                x_val, t_val = val_data
                y_val = self(x_val)
                history['val_loss'].append(self.loss(y_val, t_val))

            print(', '.join('%s = %.2f' % (k, v[-1])
                            for k, v in history.items() if v))
            
            for callback in callbacks:
                callback(self)

        return history
            
    @abstractmethod
    def forward(self, input):
        """Receive the input and compute the output.

        Parameters should not be updated in this method.
        
        Args:
            input: an input vector or matrix

        Returns:
            An output vector or matrix.
        """
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, output, target):
        """Back-propagate the error between the output and the target.

        Gradients are computed and stored for each parameter.

        Returns:
            The loss of the output with regard to the target.
        """
        raise NotImplementedError
    
    def __call__(self, input):
        return self.forward(input)

    @staticmethod
    def loss(output, target, metric='l2', average=True):
        target = assure_2D(target)

        if metric == 'l2':
            L = np.sum((output - target) ** 2)
        elif metric == 'l1':
            L = np.sum(np.abs(output - target))
        else:
            raise ValueError('unknown loss metric')

        if average: L /= len(output)
        return L


class Sequential(NN):
    """Sequential neural network."""

    def __init__(self, input_dim, *layers, activation=None):
        """Construct a sequential neural network.

        Args:
            input_dim: dimension of input
            layers: a sequence of layers. A dense layer can be input as an int which is its size.
            activation (optional): the default activation of layers
        """
        super().__init__()
        self.shape = [input_dim]
        self.layers = []
        self.activation = activation
        for layer in layers:
            self.add(Dense(layer) if is_int(layer) else layer)
        
    @property
    def depth(self):
        return len(self.layers)
        
    def add(self, layer):
        layer.init(self.shape[-1])
        
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

    def backward(self, output, target):
        error = output - target
        for k in reversed(range(self.depth)):
            error = self.layers[k].backward(error, pass_error=bool(k))
            if error is None: break
            
    def step(self, input, target):
        output = self.forward(input)
        self.backward(output, target)

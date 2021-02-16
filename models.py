import numpy as np
import pickle
from optimizers import Optimizer
from functions import Loss, ActivationAccess
from layers import Layer, Dense
from utils import *


class NN(baseclass):
    """Base class of a neural network."""
    
    def __init__(self):
        self.parameters = []
    
    def fit(self, input, target, *, epochs=20, lr=None, bs=None,
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
            A dict of training history including loss etc.
        """
        input, target = reshape2D(input), reshape2D(target)
        
        optimizer = Optimizer.get(optimizer, lr)
        loss_func = Loss.get(loss)
        
        batches = BatchLoader(input, target, batch_size=bs)
        history = {'loss': [], 'val_loss': []}

        print('\nStart training', self)
        print('Input shape:', input.shape)
        print('Target shape:', target.shape)
        print('Total epochs:', epochs)
        print('Batch size:', batches.batch_size)
        print('Optimizer:', optimizer)

        for epoch in range(epochs):
            print('\nEpoch:', epoch + 1)
            
            loss = 0
            for xb, tb in pbar(batches):
                yb = self.forward(xb)               # forward pass the input
                loss += loss_func(yb, tb)           # accumulate the loss of the output
                eb = loss_func.backward(yb, tb)     # the error in the output layer
                self.backward(eb)                   # backprop the error
                optimizer.update(self.parameters)   # update parameters

            history['loss'].append(loss / len(target))
            
            if val_data:
                x_val, t_val = val_data
                y_val = self(x_val)
                history['val_loss'].append(loss_func(y_val, t_val))

            print('\t' + ', '.join('%s = %.2f' % (k, v[-1])
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
    def backward(self, error):
        """Back-propagate the error between the output and the target.

        Gradients are computed and stored for each parameter.

        Returns:
            The loss of the output with regard to the target.
        """
        raise NotImplementedError
    
    def __call__(self, input):
        return self.forward(input)

    def loss(self, input, target, loss: Union[Loss, str] = 'l2'):
        """Compute the average loss given the input and the target data."""
        return Loss.get(loss)(self.forward(input), target) / len(input)

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
    activation = ActivationAccess()

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
            self.add(Dense(layer)
                     if isinstance(layer, numbers.Integral)
                     else layer)
        
    @property
    def depth(self):
        return len(self.layers)
        
    def add(self, layer: Layer):
        if layer._built:
            assert layer.input_dim == self.shape[-1], \
                "cannot match the input dimensionality of %s" % layer
        else:
            layer.setup(self.shape[-1])
        
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
    
    
class Convolutional(NN):
    """Convolutional neural network."""
    
    
class Recurrent(NN):
    """Recurrent neural network."""

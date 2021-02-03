import numpy as np
import matplotlib.pyplot as plt
import time
from optimizers import *
from activations import *
from utils import *


class NN:
    """Base class of a neural network"""
    
    def __init__(self):
        self.parameters = []
    
    def fit(self, input, target, epochs=20, batch_size=32, optimizer=SGD(),
            val_data=None, callbacks=()):
        """
        Given the input data, train the parameters to fit the target data.
        
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
            
    def forward(self, input):
        """
        Receive the input and compute the output.
        
        Args:
            input: an input vector or matrix

        Returns:
            an output vector or matrix
        """
        raise NotImplementedError
    
    def backward(self, output, target):
        """
        Backpropagate the error between the output and the target.

        Returns:
            the loss of the output with regard to the target
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
    """Sequential neural network"""

    def __init__(self, input_dim, *layers, activation=None):
        """
        Construct a sequential neural network

        Args:
            input_dim: dimension of input
            layers: a sequence of layers; a dense layer can be input as an int which is its size
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

        if layer.activation is None:
            layer.activation = self.activation
    
    def forward(self, input, start_layer=0):
        for k in range(start_layer, self.depth):
            output = self.layers[k].forward(input)
            input = output
        return output

    def backward(self, output, target):
        error = output - target
        for k in reversed(range(self.depth)):
            error = self.layers[k].backward(error, pass_error=k)
            if error is None: break
            
    def step(self, input, target):
        output = self.forward(input)
        self.backward(output, target)

    
class Layer:
    def __init__(self, size, *, activation=None):
        self.size = size
        self.parameters = {}
        self.input = None
        self.output = None
        self._sigma = None
        self.activation = activation
    
    @property
    def activation(self):
        return self._sigma
        
    @activation.setter
    def activation(self, sigma):
        if type(sigma) is str:
            sigma = sigma.lower()
            if sigma == 'tanh':
                self._sigma = Tanh()
            elif sigma == 'logistic':
                self._sigma = Logistic()
            elif sigma == 'relu':
                self._sigma = ReLU()
            else:
                raise ValueError('unknown activation function')
        elif isinstance(sigma, Activation) or sigma is None:
            self._sigma = sigma
        else:
            raise ValueError('unknown activation function')
        
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.parameters[name] = value
        elif hasattr(self, 'parameters') and name in self.parameters:
            del self.parameters[name]
        super().__setattr__(name, value)
        
    def init(self, input_dim):
        """
        Initialize the weights.
        """
        raise NotImplementedError
    
    def forward(self, input):
        """
        Take the input and compute the output.
        """
        raise NotImplementedError

    def backward(self, output, target, pass_error=True):
        """
        Back prop the error and record the gradient.

        Returns:
            the error passed to the previous layer
        """
        raise NotImplementedError
    
    def __call__(self, input):
        return self.forward(input)
        
        
class Dense(Layer):
    def __init__(self, size, *, with_bias=True):
        super().__init__(size)
        self.with_bias = with_bias
        
    def init(self, input_dim):
        self.weights = Parameter.rand_init(input_dim + self.with_bias, self.size)
        
    def forward(self, x, start_layer=0):
        if len(np.shape(x)) == 1:  # a single vector
            x = np.array([x])
            batch = False
        else:  # a batch of vectors
            batch = True
            
        if self.with_bias:
            b, w = self.weights[0], self.weights[1:]
            y = x @ w + b
        else:
            y = x @ self.weights
            
        if self.activation:
            y = self.activation(y)
            
        if not batch: y = y[0]

        self.input = x
        self.output = y
        return y

    def backward(self, e, pass_error=True):
        if self.output is None:
            return  # has not passed forward new data

        if self.activation:
            e *= self.activation.deriv(self.output)
            
        grad = self.input.T @ e
        if self.with_bias:  # insert the gradient of bias
            grad_b = np.sum(e, axis=0)
            grad = np.insert(grad, 0, grad_b, axis=0)
        self.weights.grad += grad
        
        # clear the input and output record
        self.input = self.output = None
        
        if pass_error:
            # return the error passed to the previous layer
            return e @ self.weights[1:].T


class Parameter(np.ndarray):
    """A trainable parameter in the neural network"""
    
    @staticmethod
    def rand_init(*size):
        """Initialize a parameter array using Xavier initialization."""
        sigma = 1 / np.sqrt(size[0])
        param = np.random.normal(scale=sigma, size=size)
        return Parameter(param)

    def __new__(cls, value):
        # convert the value to an array
        param = np.asarray(value).view(cls)
        # gradient of the parameter
        param.grad = 0
        # record of the last update made by the optimizer
        param.delta = 0
        # return the new parameter
        return param
    
    def zero_grad(self):
        self.grad = 0
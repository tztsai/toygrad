import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import random
from utils import *
from activations import *


class Sequential:
    """Basic Sequential Neural Network"""
    learning_rate = 1e-3
    momentum = 0.6

    def __init__(self, input_dim, *layers, lr=learning_rate, 
                 momentum=momentum, activation=None):
        """
        Construct a neural network

        Args:
            shape (list of int): number of nodes in each layer
            lr (optional): learning rate
        """
        self.shape = (input_dim,)
        self.layers = ()
        self.lr = lr
        self.momentum = momentum
        for layer in layers:
            self.add(layer, activation)
        
    @property
    def depth(self):
        return len(self.layers)
        
    def add(self, layer, activation=None):
        layer.init(self.shape[-1])
        self.layers += (layer,)
        self.shape += (layer.size,)
        if activation is not None:
            layer.activation = activation
    
    def forward(self, input, start_layer=0):
        """
        Pass forward the input to produce the output.

        Args:
            input: an input vector or matrix
            start_layer: the layer to feed the input

        Returns:
            an output vector or matrix
        """
        if len(np.shape(input)) == 1:  # a single vector
            input = np.array([input])
            batch = False
        else:  # a batch of vectors
            batch = True
            
        for k in range(start_layer, self.depth):
            output = self.layers[k].forward(input)
            input = output
        
        return output if batch else output[0]

    def backward(self, output, labels):
        """
        Backprop the error between the output and the target.
        """
        error = output - labels
        for k in reversed(range(self.depth)):
            error = self.layers[k].backward(error, pass_error=k)
            if error is None: break

    def update(self):
        for layer in self.layers:
            layer.update(self.lr, self.momentum)

    def fit(self, input, target, epochs=20, val_data=None,
            plot_curve=True, callbacks=()):
        if len(target.shape) == 1:
            target = target.reshape(-1, 1)

        assert input.shape[1] == self.shape[0], 'input dimension mismatch'
        assert target.shape[1] == self.shape[-1], 'output dimension mismatch'

        batches = BatchLoader(input, target)
        history = {'loss': [], 'val_loss': []}

        for epoch in range(epochs):
            print('\nEpoch:', epoch + 1)
            loss = 0

            for xb, tb in pbar(batches):
                # compute output
                yb = self.forward(xb)

                # backprop error
                self.backward(yb, tb)

                # update weights
                self.update()
                
                # update loss
                loss += np.sum((yb - tb) ** 2)

            history['loss'].append(loss / len(input))
            if val_data:
                history['val_loss'].append(self.loss(val_data[0], val_data[1]))

            print(', '.join('%s = %.2f' % (k, v[-1])
                            for k, v in history.items() if len(v) > 0))
            
            for callback in callbacks:
                callback(self)

        if plot_curve:
            fig, ax = plt.subplots()
            plot_history(history['loss'], label='Loss', ax=ax)
            if val_data:
                plot_history(history['val_loss'], label='Validation loss', ax=ax)
            ax.legend()
            plt.show()
            
    
class Layer:
    def __init__(self, size, *, activation=None):
        self.size = size
        self.weights = None
        self._input = None
        self._output = None
        self._grad = 0
        self._delta = 0
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
        Back prop the error to compute the gradient.

        Returns:
            the error passed to the previous layer
        """
        raise NotImplementedError
    
    def __call__(self, input):
        return self.forward(input)
    
    def update(self, eta, alpha):
        """
        Update weights.
        
        Args:
            eta: learning rate
            alpha: momentum
        """
        delta = alpha * self._delta - (1 - alpha) * self._grad
        self.weights += eta * delta
        self._delta = delta
        self._grad = 0
        
        
class Dense(Layer):
    def __init__(self, size, *, with_bias=True):
        super().__init__(size)
        self.with_bias = with_bias
        
    def init(self, input_dim):
        self.weights = init_weight(input_dim + self.with_bias, self.size)
        
    def forward(self, x, start_layer=0):
        self._input = x
        
        if self.with_bias:
            b, w = self.weights[0], self.weights[1:]
            y = x @ w + b
        else:
            y = x @ w
            
        if self.activation:
            y = self.activation(y)
            
        self._output = y
        return y

    def backward(self, e, pass_error=True):
        if self._output is None:  # has not passed forward new data
            return

        if self.activation:
            e *= self.activation.deriv(self._output)
            
        grad = self._input.T @ e
        if self.with_bias:  # insert the gradient of bias
            db = np.sum(e, axis=0)
            grad = np.insert(grad, 0, db, axis=0)
        self._grad += grad
        
        # clear the input and output record
        self._input = self._output = None
        
        if pass_error:
            # return the error passed back to the previous layer
            return e @ self.weights[1:].T


import numpy as np
from activations import *
from functools import wraps
from utils import baseclass, abstractmethod


class Parameter(np.ndarray):
    """A trainable parameter in the neural network."""

    @staticmethod
    def rand_init(*size, scale=None):
        """Randomly initialize a parameter array, following normal distribution.
        
        Args:
            size (int/list): shape of the parameter
            scale (optional): the standard deviation of the normal distribution,
                by default sqrt(size[0])^-1 (cf. Xavier initialization)
        """
        sigma = 1 / np.sqrt(size[0]) if scale is None else scale
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

    def need_update(self):
        return type(self.grad) is not int or self.grad != 0

    
class Layer(baseclass):
    """Base class of a neural network layer."""
    
    def __init__(self, size, *, activation=False):
        """Initialize a layer of neurons.
        
        Args:
            size: number of neurons in this layer
            activation (optional): Activation function of the layer's output.
                Its type should be str, Activation or None. None means no activation.
        """
        self.size = size
        self.parameters = {}
        self.input = None
        self.output = None
        self._sigma = None
        self.activation = activation

        self.__wrap_forward()
        self.__wrap_backward()
    
    @property
    def activation(self):
        return self._sigma
        
    @activation.setter
    def activation(self, sigma):
        if type(sigma) is str:  # interpret the str
            sigma = sigma.lower()
            if sigma == 'tanh':
                self._sigma = Tanh()
            elif sigma == 'logistic':
                self._sigma = Logistic()
            elif sigma == 'relu':
                self._sigma = ReLU()
            elif sigma == 'linear':
                self._sigma = None
            else:
                raise ValueError('unknown activation function')
        elif isinstance(sigma, Activation) or not sigma:
            self._sigma = sigma
        else:
            raise ValueError('unknown activation function')
        
    def __setattr__(self, name, value):
        # if an attribute is a Parameter, auto add to self.parameters
        if isinstance(value, Parameter):
            self.parameters[name] = value
        # if a member of self.parameters is set to a non-Parameter, remove it
        elif hasattr(self, 'parameters') and name in self.parameters:
            del self.parameters[name]
        super().__setattr__(name, value)
        
    @abstractmethod
    def init(self, input_dim):
        """Initialize the weights."""
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, input):
        """Take the input and compute the output."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, error, pass_error: bool = True):
        """Back prop the error and record the gradients of parameters.

        Returns:
            The error passed to the previous layer, if `pass_error` is True.
        """
        raise NotImplementedError
    
    def __wrap_forward(self):
        """Wraps common procedures around the forward method."""
        forward = self.forward
        
        @wraps(forward)
        def wrapped(input):
            if len(np.shape(input)) == 1:  # a single vector
                input = np.array([input])
                batch = False
            else:  # a batch of vectors
                batch = True
                
            # call the subclass forward method
            output = forward(input)
            
            # record input and output
            self.input, self.output = input, output
            
            if self.activation:
                output = self.activation(output)
            if not batch:
                output = output[0]
            return output
        
        self.forward = wrapped
    
    def __wrap_backward(self):
        """Wraps common procedures around the backward method."""
        backward = self.backward
        
        @wraps(backward)
        def wrapped(error, **kwds):
            # has not passed forward new input
            if self.output is None:
                return None
            
            # backprop the error through the activation
            if self.activation:
                error *= self.activation.deriv(self.output)
            
            # call the subclass backward method
            error = backward(error, **kwds)
            
            # clear input and output records
            self.input = self.output = None
            
            return error
        
        self.backward = wrapped
    
    def __call__(self, input):
        return self.forward(input)
        
        
class Dense(Layer):
    """Dense or fully-connected layer."""
    
    def __init__(self, size, with_bias=True, **kwds):
        super().__init__(size, **kwds)
        self.with_bias = with_bias
        
    def init(self, input_dim):
        self.weights = Parameter.rand_init(input_dim + self.with_bias, self.size)
        
    def forward(self, input):
        if self.with_bias:
            bias, weights = self.weights[0], self.weights[1:]
            return input @ weights + bias
        else:
            return input @ self.weights

    def backward(self, error, pass_error=True):
        grad = self.input.T @ error
        if self.with_bias:  # insert the gradient of bias
            grad_b = np.sum(error, axis=0)
            grad = np.insert(grad, 0, grad_b, axis=0)
        self.weights.grad += grad

        if pass_error:
            # return the error passed to the previous layer
            return error @ self.weights[1:].T
        
        
class RBF(Layer):
    """Radial-basis function layer."""
    sigma = 1
    step_size = 0.02  # scaling factor of the update of an RBF center
    neighborhood_radius = 0.03  # nodes within the neighborhood will update with the winner

    def init(self, input_dim):
        self.centers = Parameter.rand_init(self.size, input_dim, scale=1)
        
    @staticmethod
    def gaussian_rbf(x, mu, sigma=sigma):
        return np.exp(-np.sum((x - mu)**2, axis=-1) / (2 * sigma**2))
        
    def forward(self, input, update_centers=True):
        return self.gaussian_rbf(input[:, None], self.centers)

    def backward(self, error, pass_error=True):
        """Apply the SOM algorithm to update parameters."""

        # for each input point, find the "winner" - the nearest RBF center
        winners = np.argmax(self.output, axis=1)

        for i, j in enumerate(winners):
            # search for RBF nodes in the neighborhood of the winner
            for k in range(self.size):
                dist = abs(self.output[i, j] - self.output[i, k])
                if dist <= self.neighborhood_radius:
                    # node k is in the neighborhood of the winner (node j)
                    delta = self.input[i] - self.centers[k]
                    # move the center nearer to the i-th input
                    self.centers[k] += self.step_size * delta

        # the RBF layer does not backprop error
        return None

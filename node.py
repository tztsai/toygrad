from function import Function
from utils import *


class Node(Function):
    """Base class of a computational node."""

    def __init__(self, dim_out=None, dim_in=None):
        """Initialize a computational node.

        Args:
            dim_out: the dimensionality of the output
            dim_in: the dimensionality of the input
        """
        self.dim_out = dim_out
        self.dim_in = dim_in
        self.output = None
        self.input = None
        self.ascendants = []
        self.descendants = []
        self.network = None  # the network containing this node

    @property
    def training(self):
        return self.network and self.network.training
        
    def connect(self, node):
        """Connect a node as a descendant."""
        self.descendants.append(node)
        node.ascendants.append(self)
        if self.network:
            self.network.add(node)

    @abstractmethod
    def setup(self):
        """Initialize the parameters if the node has any."""
        if all(type(node.dim_out) is int for node in self.ascendants):
            dim_in = sum(node.dim_out for node in self.ascendants)
            if self.dim_in is None:
                self.dim_in = dim_in
            elif self.dim_in != dim_in:
                raise AssertionError('node dimensionalities mismatch')
        print(f"Setup {self}.")
        
    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError

    @abstractmethod
    def backward(self, *error):
        raise NotImplementedError

    def _wrapped_forward(self, *input):
        # call the subclass forward method
        forward = super().__getattribute__("forward")
        output = forward(*input)
        
        

        # record input and output
        self.input, self.output = input, output
        return output

    def _wrapped_backward(self, error):
        # has not passed forward new input
        if self.output is None: return

        # call the subclass backward method
        backward = super().__getattribute__("backward")
        error = backward(error, pass_error)

        # clear input and output records
        self.input = self.output = None

        return error

    def __setattr__(self, name, value):
        if not hasattr(self, 'parameters'):
            super().__setattr__('parameters', {})

        # if an attribute is a Parameter, auto add to self.parameters
        if isinstance(value, Parameter):
            self.parameters[name] = value
        # if a member of self.parameters is set to a non-Parameter, remove it
        elif name in self.parameters:
            del self.parameters[name]

        super().__setattr__(name, value)

    def __repr__(self):
        return (f"{type(self).__name__}({self.size}%s%s)" %
                ("" if not self.activation else
                 ", activation='%s'" % type(self.activation).__name__,
                 "" if not self.dropout else
                 ", dropout=%f" % self.dropout.p))


class Parameter(np.ndarray):
    """A trainable parameter in a node."""

    def __new__(cls, value=None, *, size=None, mean=0, scale=None):
        """Create a new Parameter.

        If `value` is given, then it will be converted to a Parameter.
        However, if `size` is additionally specified, then a new Parameter
        of this size will be created filled with the given `value`.
        If `value` is not given, then `size` must be provided to generate
        a random Parameter following Gaussian distribution. Additionally,
        `mean` and `scale` of the Gaussian can be specified.
        """
        if value is None:  # random initialization
            if size is None:
                raise AssertionError('the size of the Parameter must be'
                                     'given for random initialization')
            if scale is None:
                length = size[0] if hasattr(size, '__len__') else size
                scale = 1 / np.sqrt(length)  # Xavier initialization
            param = np.random.normal(loc=mean, scale=scale, size=size)
            return Parameter(param)
        else:  # convert the value to an array
            if size is not None:  # fill an array of the given size
                value = np.full(size, value)
            return np.asarray(value).view(cls)

    @property
    def grad(self):
        if not hasattr(self, '_grad'):
            self._grad = 0
            self._dirty = False
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value
        self._dirty = True

    def zero_grad(self):
        self._grad = 0
        self._dirty = False

    @property
    def need_update(self):
        return self._dirty


class Compose(Node):
    
    def __init__(self, *nodes):
        """Composes several nodes into one node."""
        self.nodes = nodes
        
    def forward(self, input):
        for node in self.nodes:
            output = node.forward(input)
            input = output
        return output
    
    def backward(self, error):
        for node in reversed(self.nodes):
            error = node.backward(error)
        return error
        

class Affine(Node):
    """Affine transformation."""

    def __init__(self, size, with_bias=True, **kwds):
        """
        Extra args:
            with_bias: whether the affine transformation has a constant bias
        """
        self.with_bias = with_bias
        super().__init__(size, **kwds)

    def setup(self, input_dim):
        super().setup(input_dim)
        # coefficients of the layer's affine transformation
        self.weights = Parameter(size=[input_dim + self.with_bias, self.size])

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

        # limit the magnitude of gradient
        self.weights.grad += grad  # np.clip(grad, -1e6, 1e6)

        if pass_error:
            # return the error passed to the previous layer
            return error @ self.weights[1:].T


class RBF(Layer):
    """Radial-basis function layer."""

    def __init__(self, size, *, sigma=0.1, move_centers=True,
                 centers_mean=0, centers_deviation=1,
                 step_size=0.01, step_decay=0.25,
                 update_neighbors=False, nb_radius=0.05, **kwds):
        """
        Extra args:
            sigma: the width of each RBF node
            move_centers: whether to update the RBF centers during training
            centers_mean: the mean value of the RBF centers
            centers_deviation: the standard deviation of the RBF centers
            step_size: scaling factor of the update of an RBF center
            step_decay: linearly decrease the step size of the neighbors to be updated s.t. the step
                size of the neighbor on the border of the neighborhood is scaled by `step_decay`
            update_neighbors: whether nodes within the winner's neighborhood will update with the winner
            nb_radius: the radius of the neighborhood
        """
        self.sigma = sigma
        self.move_centers = move_centers
        self.centers_mean = centers_mean
        self.centers_dev = centers_deviation
        self.step_size = step_size
        self.step_decay = step_decay
        self.update_neighbors = update_neighbors
        self.nb_radius = nb_radius
        super().__init__(size, **kwds)

    def setup(self, input_dim):
        super().setup(input_dim)

        # randomly initialize RBF centers
        self.centers = Parameter(size=[self.size, input_dim],
                                 mean=self.centers_mean,
                                 scale=self.centers_dev)

        # compute the pair-wise distances between RBF centers
        self._center_dist = {(i, j): self.distance(mu_i, mu_j)
                             for i, mu_i in enumerate(self.centers)
                             for j, mu_j in enumerate(self.centers)
                             if i < j}

    def center_dist(self, i, j):
        return self._center_dist[min(i, j), max(i, j)] if i != j else 0

    def neighborhood(self, i):
        "RBF nodes in the neighborhood of node i, including itself."
        return [j for j in range(self.size)
                if self.center_dist(i, j) < self.nb_radius]

    @staticmethod
    def distance(x, y):
        return np.sum(np.abs(x - y), axis=-1)

    def gaussian_rbf(self, x, mu):
        return np.exp(-self.distance(x, mu)**2 / (2 * self.sigma**2))

    def forward(self, input):
        """Compute the output and update RBF centers w.r.t the input."""
        output = self.gaussian_rbf(input[:, None], self.centers)

        if not self.training or not self.move_centers:
            return output  # does not move RBF centers

        # for each input point, find the "winner" - the nearest RBF center
        winners = np.argmax(output, axis=1)

        # find the neighbors of each winner
        neighbors = {j: self.neighborhood(j) for j in set(winners)}

        # gather all RBF nodes to be updated
        updated = set.union(neighbors.values())

        for i, j in enumerate(winners):
            for k in neighbors[j]:
                # move the center k nearer to the i-th input
                delta = self.input[i] - self.centers[k]
                decay = 1 - ((1 - self.step_decay) / self.nb_radius *
                             self.center_dist(j, k))
                self.centers[k] += self.step_size * decay * delta

        # update center distances
        for i, mu_i in enumerate(self.centers):
            for j, mu_j in enumerate(self.centers):
                if i < j and (i in updated or j in updated):
                    self._center_dist[i, j] = self.distance(mu_i, mu_j)

        return output

    def backward(self, error, pass_error=True):
        """The RBF layer does not compute gradients or backprop errors."""
        return

    def __repr__(self):
        repr = super().__repr__()
        return repr[:-1] + ", sigma=%.2f" % self.sigma + ")"


class Dropout(Node):
    """Randonly deactivates some neurons to mitigate overfitting."""

    def __init__(self, p, size):
        super().__init__()
        self.p = p
        self.size = size
        self._mask = None

    def forward(self, x):
        mask = bernoulli(self.size, 1 - self.p) / (1 - self.p)
        self._mask = mask
        return mask * x

    def backward(self, err):
        return self._mask * err

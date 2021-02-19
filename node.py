from function import Function
from param import Parameter
from utils import *
from devtools import *


class Node(Function):
    """Base class of a computational node."""

    def __init__(self, dim_out=None, dim_in=None):
        """Initialize a computational node.

        Args:
            dim_out: the dimensionality of the output
            dim_in: the dimensionality of the input
        """
        super().__init__()
        self.dim_out = dim_out
        self.dim_in = dim_in
        self.output = None
        self.input = None
        self.ascendants = []
        self.descendants = []
        self.network = None  # the network containing this node
        self._has_setup = False
        
        if dim_out and dim_in:
            self.setup(dim_in)

    @property
    def training(self):
        return self.network and self.network.training
        
    def connect(self, node):
        """Connect a node as a descendant."""
        self.descendants.append(node)
        node.ascendants.append(self)
        if self.network:
            self.network.add(node)
            
    @contextmanager
    def isolate(self):
        asc = self.ascendants
        desc = self.descendants
        self.ascendants = []
        self.descendants = []
        yield  # do sth with the node here
        self.ascendants = asc
        self.descendants = desc
        
    @abstractmethod
    def setup(self, dim_in=None):
        """Initialize the parameters if the node has any."""
        if self._has_setup:
            dbg(f'{self} has been setup')
            return
            
        if dim_in is None:
            dim_in = [node.dim_out for node in self.ascendants]
            assert dim_in, f'input dimensionality of {self} not given'
            
        if dim(dim_in) == 0:
            self.dim_in = dim_in
            if self.ascendants:
                if len(self.ascendants) > 1:
                    raise AssertionError(f'dimensionality mismatch in {self}')
                elif self.ascendants[0].dim_out:
                    pass
        elif all(type(d) is int for d in dim_in):
            if len(dim_in) == 1:
                dim_in = dim_in[0]
            if self.dim_in is None:
                self.dim_in = dim_in
            elif self.dim_in != dim_in:
                raise AssertionError(f'dimensionality mismatch in {self}')
        else:
            raise AssertionError(f'input dimension of {self} not given')
        
        self._has_setup = True
        info(f"Setup {repr(self)}.")
        
    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError

    @abstractmethod
    def backward(self, *error):
        raise NotImplementedError
    
    def match_io(self, nodes, data):
        if not nodes:
            return ()
        elif len(nodes) == 1:
            return nodes[0], data
        else:
            if type(data) is not tuple or len(data) != len(nodes):
                raise TypeError(f"I/O mismatch in {self}")
            return zip(nodes, data)

    def _wrapped_forward(self, *input):
        # call the inner forward method
        output = self._forward(*input)
        
        # record input and output
        if len(input) == 1: input = input[0]
        self.input, self.output = input, output
        
        # pass forward the output
        for node, out in self.match_io(self.descendants, output):
            node.forward(out)
            
        return output

    def _wrapped_backward(self, *error):
        # has not passed forward new input
        if self.output is None: return
        
        # backward is blocked by a descendant
        if any(e is None for e in error): return

        # call the inner backward method
        error = self._backward(*error)

        # clear input and output records
        self.input = self.output = None

        # pass backward the error
        for node, err in self.match_io(self.ascendants, error):
            node.backward(err)
            
        return error

    def __setattr__(self, name, value):
        "Automatically keeps track of trainable parameters."
        if not hasattr(self, 'parameters'):
            super().__setattr__('parameters', {})

        # if an attribute is a Parameter, auto add to self.parameters
        if isinstance(value, Parameter):
            self.parameters[name] = value
        # if a member of self.parameters is set to a non-Parameter, remove it
        elif name in self.parameters:
            del self.parameters[name]

        super().__setattr__(name, value)
        
    def __str__(self):
        return f'node({id(self)})'

    def __repr__(self):
        return (f"{type(self).__name__}({self.size}%s%s)" %
                ("" if not self.activation else
                 ", activation='%s'" % type(self.activation).__name__,
                 "" if not self.dropout else
                 ", dropout=%f" % self.dropout.p))


class Compose(Node):
    
    def __init__(self, *nodes):
        """Composes several nodes into one node."""
        assert nodes, "no nodes are composed"
        
        super().__init__(nodes[0].dim_in, nodes[-1].dim_out)
        
        self.nodes = nodes
        for i in range(len(nodes) - 1):
            nodes[i].connect(nodes[i + 1])
            
    @property
    def dim_in(self):
        return self.nodes[0].dim_in
    
    @property
    def dim_out(self):
        return self.nodes[-1].dim_out
            
    def setup(self):
        for node in self.nodes:
            node.setup()
        
    def forward(self, input):
        for node in self.nodes:
            output = node.forward(input)
            input = output
        return output
    
    def backward(self, error):
        for node in reversed(self.nodes):
            error = node.backward(error)
        return error
    
    def __repr__(self):
        node_list = ', '.join(map(repr, self.nodes))
        return f'Compose({node_list})'
        

class Affine(Node):
    """Affine transformation."""

    def __init__(self, dim_out, with_bias=True, dim_in=None):
        """
        Extra args:
            with_bias: whether the affine transformation has a constant bias
        """
        super().__init__(dim_out, dim_in)
        self.with_bias = with_bias

    def setup(self):
        super().setup()
        # coefficients of the affine transformation
        self.weights = Parameter(size=[self.dim_in + self.with_bias, self.dim_out])

    def forward(self, input):
        if self.with_bias:
            bias, weights = self.weights[0], self.weights[1:]
            return input @ weights + bias
        else:
            return input @ self.weights

    def backward(self, error):
        if self.training:
            grad = self.input.T @ error
            
            if self.with_bias:  # insert the gradient of bias
                grad_b = np.sum(error, axis=0)
                grad = np.insert(grad, 0, grad_b, axis=0)

            self.weights.grad += grad

        if self.ascendants:
            # return the error passed to the previous layer
            return error @ self.weights[1:].T


class RBF(Node):
    """Radial-basis function."""

    def __init__(self, size, *, sigma=0.1, move_centers=True,
                 update_neighbors=False, nb_radius=0.05,
                 step_size=0.01, step_decay=0.25):
        """
        Args:
            size: the number of RBF units in the node, i.e. the dim of output
            sigma: the width of each RBF unit
            move_centers: whether to update the RBF centers during training
            update_neighbors: whether units within the winner's neighborhood will update with the winner
            nb_radius: the radius of the neighborhood
            step_size: scaling factor of the update of an RBF center
            step_decay: linearly decrease the step size of the neighbors to be updated s.t. the step
                size of the neighbor on the border of the neighborhood is scaled by `step_decay`
        """
        super().__init__(size)
        self.sigma = sigma
        self.move_centers = move_centers
        self.step_size = step_size
        self.step_decay = step_decay
        self.update_neighbors = update_neighbors
        self.nb_radius = nb_radius

    def setup(self, *, centers=None, mean=0, stddev=1):
        super().setup()

        if centers is not None:
            assert np.shape(centers) == (self.dim_out, self.dim_in), \
                "the shape of the centers mismatches the RBF node"
            self.centers = Parameter(centers)
        else:
            # randomly initialize RBF centers
            self.centers = Parameter(size=[self.dim_out, self.dim_in],
                                     mean=mean, scale=stddev)

        # compute the pair-wise distances between RBF centers
        self._center_dist = {(i, j): self.distance(mu_i, mu_j)
                             for i, mu_i in enumerate(self.centers)
                             for j, mu_j in enumerate(self.centers)
                             if i < j}

    def center_dist(self, i, j):
        return self._center_dist[min(i, j), max(i, j)] if i != j else 0

    def neighborhood(self, i):
        "RBF units in the neighborhood of unit i, including itself."
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

        # gather all RBF units to be updated
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

    def backward(self, error):
        """The RBF layer does not compute gradients or backprop errors."""
        return

    def __repr__(self):
        repr = super().__repr__()
        return repr[:-1] + ", sigma=%.2f" % self.sigma + ")"


class Dropout(Node):
    """Randonly deactivates some neurons to mitigate overfitting."""

    def __init__(self, size, p):
        super().__init__(size)
        self.p = p
        self.size = size
        self._mask = None

    def forward(self, x):
        mask = bernoulli(self.size, 1 - self.p) / (1 - self.p)
        self._mask = mask
        return mask * x

    def backward(self, err):
        return self._mask * err

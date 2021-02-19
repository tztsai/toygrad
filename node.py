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
        self.training = True
        
        self._forward = super().__getattribute__("forward")
        self._backward = super().__getattribute__("backward")
        
        self._has_setup = False

    # @property
    # def _root(self):
    #     if self.__root is None:
    #         return None
    #     else:
    #         while (r := self.__root._root) is not None:
    #             self.__root = r
    #         return self.__root

    # @_root.setter
    # def _root(self, root):
    #     self.__root = root
        
    def connect(self, node):
        """Connect a node as a descendant."""
        self.descendants.append(node)
        node.ascendants.append(self)
        # node._root = self._root
        return node
            
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
            
        asc_dim_out = [node.dim_out for node in self.ascendants]
        
        if dim_in is None:
            dim_in = self.dim_in
            
        if dim_in is None:
            dim_in = asc_dim_out
            assert dim_in, f'dimensionality unspecified in {self}'
        
        if self.ascendants:
            for k, (do, di) in enumerate(
                self._match_dim(asc_dim_out, dim_in)):
                    if do is None:
                        self.ascendants[k].dim_out = di
                    else:
                        dim_in[k] = do
            
        self.dim_in = squeeze(dim_in)
        self._has_setup = True
        
    def _match_dim(self, d1, d2):
        if dim(d1) == 0: d1 = [d1]
        if dim(d2) == 0: d2 = [d2]
        
        assert len(d1) == len(d2), f'dimensionality mismatch in {self}'
        
        for x, y in zip(d1, d2):
            if x is None and y is None:
                raise AssertionError(f'dimensionality unspecified in {self}')
            elif (x is None) ^ (y is None):
                yield x, y  # set the unspecified dim to the other one
            else:
                assert x == y, f'dimensionality mismatch in {self}'
    
    def _match_pass(self, nodes, data):
        if not nodes:  # no nodes to pass
            return ()
        
        if len(nodes) == 1:
            data = data,
            
        if type(data) is not tuple or len(data) != len(nodes):
            raise AssertionError(f"arity mismatch in the forward/backward pass of {self}")
        
        return zip(nodes, data)
        
    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError

    @abstractmethod
    def backward(self, *error):
        raise NotImplementedError
    
    def state(self):
        return self.__dict__.copy()
        
    def __getattribute__(self, attr):
        if attr == "forward":
            return super().__getattribute__("_wrapped_forward")
        elif attr == "backward":
            return super().__getattribute__("_wrapped_backward")
        else:
            return super().__getattribute__(attr)

    def _wrapped_forward(self, *input):
        """Wrapper of forward method to add preprocessing and postprocessing.
        
        When the user calls `forward`, `_wrapped_forward` will be called instead.
        The `forward` method implemented by a subclass is accessed via `self._forward`.
        """
        # call the inner forward method
        output = self._forward(*input)
        
        # record input and output
        input = squeeze(input)
        self.input, self.output = input, output
        
        # pass forward the output
        for node, out in self._match_pass(self.descendants, output):
            node.forward(out)
            
        return output

    def _wrapped_backward(self, *error):
        """Wrapper of backward method to add preprocessing and postprocessing."""
        # has not passed forward new input
        if self.output is None: return
        
        # backward is blocked by a descendant
        if any(e is None for e in error): return

        # call the inner backward method
        error = self._backward(*error)

        # clear input and output records
        self.input = self.output = None

        # pass backward the error
        for node, err in self._match_pass(self.ascendants, error):
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
        return f'node({hex(id(self))[-3:]})'

    def __repr__(self):
        nodetype = type(self).__name__
        return nodetype + f'({self.dim_in}, {self.dim_out})'


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
            
    def setup(self, dim_in=None):
        for i, node in enumerate(self.nodes):
            node.setup(None if i else dim_in)
        
    def forward(self, *input):
        for node in self.nodes:
            output = node.forward(*input)
            input = output
        return output
    
    def backward(self, *error):
        for node in reversed(self.nodes):
            error = node.backward(error)
        return error
    
    def __repr__(self):
        node_list = ', '.join(map(repr, self.nodes))
        return f'Compose({node_list})'
        

class Affine(Node):
    """Affine transformation."""

    def __init__(self, dim_out, dim_in=None, *, with_bias=True):
        """
        Args:
            with_bias: whether the affine transformation has a constant bias
        """
        super().__init__(dim_out, dim_in)
        self.with_bias = with_bias

    def setup(self, dim_in=None):
        super().setup(dim_in)
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

    def __init__(self, size, dim, *, sigma=0.1, move_centers=True,
                 update_neighbors=False, nb_radius=0.05,
                 step_size=0.01, step_decay=0.25):
        """
        Args:
            size: the number of RBF units in the node, i.e. the dim of output
            dim: the dim of each RBF unit, i.e. the dim of input
            sigma: the width of each RBF unit
            move_centers: whether to update the RBF centers during training
            update_neighbors: whether units within the winner's neighborhood will update with the winner
            nb_radius: the radius of the neighborhood
            step_size: scaling factor of the update of an RBF center
            step_decay: linearly decrease the step size of the neighbors to be updated s.t. the step
                size of the neighbor on the border of the neighborhood is scaled by `step_decay`
        """
        super().__init__(size, dim)
        self.sigma = sigma
        self.move_centers = move_centers
        self.step_size = step_size
        self.step_decay = step_decay
        self.update_neighbors = update_neighbors
        self.nb_radius = nb_radius

    def setup(self, dim=None, *, centers=None, mean=0, stddev=1):
        super().setup(dim)

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


# %% test
if __name__ == '__main__':
    a = Affine(5)
    b = Affine(8)
    a.connect(b)
    a.setup(3)
    a(np.ones([10, 3]))

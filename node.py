# %% Computational Nodes

from function import Function
from param import Parameter
from utils import *
from devtools import *
from my_utils.utils import main


NODES = {}


def get_node(val):
    """Converts integers or strings to nodes."""
    if isinstance(val, int) and val > 0:
        return Linear(val)  # interger -> Linear node
    elif val == 'tanh':
        return Tanh
    elif val in ['logistic', 'sigmoid']:
        return Logistic
    elif val == 'relu':
        return ReLU
    elif val == 'linear':
        return None
    elif type(val) is str and val in NODES:
        return NODES[val]
    else:
        raise ValueError(f"unknown node: {val}")


class Node(Function, metaclass=makemeta(get_node)):
    """Base class of a computational node."""
    
    def __new__(cls):
        node = object.__new__(cls)
        node.id = hex(id(node))[-3:]
        while node.id in NODES:
            node.id = hex(rand.randint(16))[-1] + node.id
        NODES[node.id] = node
        return node
    
    def __init__(self, fan_out=None, fan_in=None):
        """Initialize a computational node.

        Args:
            fan_out: the shape of the output
                if there are multiple descendants, its shape should be a list, otherwise it can
                be an integer, a tuple (when multi-dim) or a list containing a single item
            fan_in: the shape of the input
                the type is the same as fan_out
        """
        super().__init__()
        
        self.ascendants = []
        self.descendants = []
        self.training = True
        
        self.output = None
        self.input = None
        # self.error = None
        
        self._fi = self._fan_io(fan_in)
        self._fo = self._fan_io(fan_out)
        
        self._forward = super().__getattribute__("forward")
        self._backward = super().__getattribute__("backward")
        
        self._has_setup = False
        self._block = False  # block forward pass to prevent looping
        
    @property
    def fan_out(self):
        return squeeze(self._fo)
    
    @property
    def fan_in(self):
        return squeeze(self._fi)
    
    @staticmethod
    def _fan_io(shapes):
        "Makes sure fan-in or fan-out is a list of shape tuples."
        
        def shapeof(x):
            if x is None:
                return None
            elif type(x) is int:
                return (x,)
            elif type(x) is tuple:
                assert all(type(y) is int for y in x)
                return x
            else:
                raise AssertionError('invalid shape of fan-in/fan-out')
        
        if isinstance(shapes, list):
            return [shapeof(x) for x in shapes]
        else:
            return [shapeof(shapes)]
            
        
    def connect(self, *nodes):
        """Connect nodes as a descendant."""
        
        assert len(self.descendants) < len(self._fo), \
            f'{self} cannot connect more descendants'
        
        for node in nodes:
            node = Node(node)
            
            if node in self.descendants:
                continue  # already connected
            
            # check fan-in and fan-out shapes
            di = len(self.descendants)  # index of this descendant
            fo = self._fo[di]           # fan-out of the connection to this descendant
            
            assert fo is not None, f'the {di}-th output shape is unspecified in {self}'
            
            ai = len(node.ascendants)   # index of self in this descendant's ascendants
            fi = node._fi[ai]           # fan-in of its connection to self
            
            if node._fi[ai] is None:
                node._fi[ai] = fo
            else:
                assert fi == fo, f'connection mismatch between {self} and {node}'
            
            self.descendants.append(node)
            node.ascendants.append(self)
            
    @contextmanager
    def isolate(self):
        """Isolate the node so that forward and backward pass will not affect its neighbors."""
        
        asc = self.ascendants
        desc = self.descendants
        self.ascendants = []
        self.descendants = []
        
        yield  # do sth with the node here
        
        self.ascendants = asc
        self.descendants = desc
        
    def setup(self):
        """Inherit from this method to initialize the trainable parameters."""
        
        if self._has_setup:
            dbg(f'{self} has already been set up.')
            return
        
        if self.descendants:
            assert len(self._fo) == len(self.descendants), \
                f'output ports of {self} not fully connected'
        
        dbg('setting up %s', self)
        
        for node in self.descendants:
            node.setup()
            
        self._has_setup = True
        
    def _match_pass(self, nodes, data):
        if nodes and len(data) != len(nodes):
            raise AssertionError("data shape mismatch when "
                                 f"passing through {self}")
        
        return zip(nodes, data)
        
    @abstractmethod
    def forward(self, *input):
        """Passes forward the input and returns the output.
        
        The input data should be in batches and `len(input)` should equal to `len(self._fi)`.
        For the i-th input, its shape should be `(batch_size, *self._fi[i])`.
        
        If the node has multiple descendants, the output data should be a tuple; otherwise
        it should be an array of shape `(batch_size, *self._fi[0])`.
        """
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

    def _wrapped_forward(self, *input, batch=True):
        """Wrapper of forward method to add preprocessing and postprocessing.
        
        When the user calls `forward`, `_wrapped_forward` will be called instead.
        The `forward` method implemented by a subclass is accessed via `self._forward`.
        """
        if self._block: return
        
        # check and reshape input
        try:
            assert len(input) == len(self._fi)
            input = tuple(np.reshape(x, [len(x) if batch else 1, *fi])
                          for x, fi in zip(input, self._fi))
        except:
            raise AssertionError(f'input data shape mismatches the fan-in of {self}')
        
        # call the inner forward method
        output = self._forward(*input)
        
        # record input and output
        self.input, self.output = squeeze(input), output
        
        # check and reshape output
        if type(output) is not tuple:
            output = output,
        if not batch:
            output = tuple(y[0] for y in output)
        
        try:
            assert len(output) == len(self._fo)
            assert all((y.shape[1:] if batch else y.shape) == fo
                       for y, fo in zip(output, self._fo))
        except:
            raise AssertionError(f'output data shape mismatches the fan-out of {self}')
        
        # block to prevent an infinite loop of forward pass
        self._block = True
        
        # pass forward the output
        for node, out in self._match_pass(self.descendants, output):
            node.forward(out)
            
        # unblock to accept the next forward pass
        self._block = False
        
        return output[0] if len(output) == 1 else output

    def _wrapped_backward(self, *error):
        """Wrapper of backward method to add preprocessing and postprocessing."""
        # has not passed forward new input
        if self.output is None: return
        
        # backward is blocked by a descendant
        if any(e is None for e in error): return
        
        # check shape
        try:
            assert len(error) == len(self._fo)
            error = tuple(e.reshape(len(e), *fo) for e, fo in zip(error, self._fo))
        except:
            raise AssertionError(f'error shape mismatches the fan-out of {self}')

        # call the inner backward method
        error = self._backward(*error)

        # clear input and output records
        self.input = self.output = None
        
        # check shape after backward pass
        if type(error) is not tuple: error = error,
        assert len(error) == len(self._fi), f'error shape mismatches the fan-in of {self}'

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
        return f'{type(self).__name__}#{self.id}'

    def __repr__(self):
        nodetype = type(self).__name__
        shapes = tuple(f'{s}={d}' for s, d in
                       zip(['in', 'out'],
                           [self.fan_in, self.fan_out]))
        return nodetype + '(%s)' % ', '.join(shapes)


class Network(Node):
    
    def __init__(self, connections: list, entry=None, exit=None):
        """Accepts a list of node connections and wrap it into a single node.
        
        Format of the connection list:
            [[<node>, <child> or <list of children>], ...] (can also be a tuple)
        """
        self.entry = entry
        self.exit = exit
        
        self.nodes = self.buildnet(connections)
        
        self.ascendants = self.entry.ascendants
        self.descendants = self.exit.descendants
        
        super().__init__(self.exit.fan_out, self.entry.fan_in)
            
    def buildnet(self, connections):
        net = {}
        ascendants = set()
        descendants = set()
        
        for node, desc in connections:
            node = Node(node)
            
            if is_seq(desc):
                node.connect(*desc)
            else:
                node.connect(desc)        
                
            net[node] = node.descendants
            
            ascendants.add(node)
            descendants.update(node.descendants)
        
        if self.entry is None:
            entries = ascendants - descendants
            assert len(entries) == 1, "cannot find the entry of %s" % self
            self.entry = entries.pop()
                    
        if self.exit is None:
            exits = descendants - ascendants
            assert len(exits) == 1, "cannot find the exit of %s" % self
            self.exit = exits.pop()
        
        return net
    
    def setup(self):
        self.entry.setup()
            
    def forward(self, *input):
        self.entry.forward(*input)
        return self.exit.output
    
    def backward(self, *error):
        self.exit.backward(*error)
        return self.entry.error
    
    def __repr__(self):
        net_rep = ', '.join('%s -> [%s]' % (repr(n), seq_repr(d)) for n, d in self.nodes.items())
        return f'Network({net_rep})'
    
    
class Sequential(Network):
    """Sequential network where each node only has at most one ascendant and one descendant."""
    
    def __init__(self, *nodes):
        nodes = list(map(Node, nodes))
        connections = list(zip(nodes, nodes[1:]))
        super().__init__(connections, nodes[0], nodes[-1])
        self.nodes = nodes

    def __repr__(self):
        return f'Sequential({seq_repr(self.nodes)})'

class Linear(Node):
    """Linear transformation (actually affine transformation if bias is nonzero)."""

    def __init__(self, fan_out, fan_in=None, *, with_bias=True):
        """
        Args:
            with_bias: whether the Linear transformation has a constant bias
        """
        super().__init__(fan_out, fan_in)
        self.with_bias = with_bias

    def setup(self):
        super().setup()
        # coefficients of the Linear transformation
        self.weights = Parameter(size=[self.fan_in + self.with_bias, self.fan_out])

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
        
        
class Conv2D(Node):
    """2D convolution transformation."""
    
    scan_step = 1
    
    def __init__(self, num_filters, fan_in=None, filter_width=2):
        super().__init__(fan_in=fan_in)
        self.nf = num_filters
        self.fw = filter_width
        
    def setup(self):
        super().setup()
        assert type(self.fan_in) is tuple and len(self.fan_in) == 2, \
            f'input dimension of {self} must be 2D'
            
        self._grid = self._make_grid()
        self._fo = [(self.nf, len(self._grid), len(self._grid[0]))]
        
        self.filters = Parameter(size=[self.nf, 2*self.fw + 1, 2*self.fw + 1])

    def convolve(self, x, filter):
        assert dim(x) == 2, 'data dimension should be 2D'
        k = self.fw
        return np.array([[np.sum(filter * x[rect]) for rect in row]
                         for row in self._grid])
        
    def forward(self, input):
        return np.array([[self.convolve(im, ft) for ft in self.filters]
                         for im in input])
        
    def _make_grid(self):
        "Makes a grid of array slices."
        h, w = self.fan_in
        k = self.fw
        return [[(slice(i-k, i+k+1), slice(j-k, j+k+1))
                 for j in range(k, w - k, self.scan_step)]
                for i in range(k, h - k, self.scan_step)]


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

    def setup(self, centers=None, mean=0, stddev=1):
        super().setup()

        if centers is not None:
            assert np.shape(centers) == (self.fan_out, self.fan_in), \
                "the shape of the centers mismatches the RBF node"
            self.centers = Parameter(centers)
        else:
            # randomly initialize RBF centers
            self.centers = Parameter(size=[self.fan_out, self.fan_in],
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

                
class Activation(Node):
    def __init__(self):
        super().__init__()
        self._fi = self._fo
        
    def __repr__(self):
        return type(self).__name__ + '()'
        
        
class Tanh(Activation):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, error):
        return error * (1 - self.output**2)


class Logistic(Activation):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, error):
        return error * self.output * (1 - self.output)


class ReLU(Activation):
    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, error):
        return error * (self.output > 0)


class SoftMax(Activation):
    def forward(self, x):
        ex = np.exp(x - np.max(x, axis=1, keepdims=True))
        return ex / np.sum(ex, axis=1, keepdims=True)

    def backward(self, error):
        # TODO: not correct?
        dp = np.sum(error * self.output, axis=1, keepdims=True)
        return (error - dp) * self.output


class Dropout(Activation):
    """Randonly deactivates some neurons to mitigate overfitting."""

    def __init__(self, p):
        super().__init__()
        self.p = p
        self._mask = None

    def forward(self, x):
        self._mask = bernoulli(self.fan_in, 1 - self.p) / (1 - self.p)
        return self._mask * x

    def backward(self, error):
        return self._mask * error

    def __repr__(self):
        return 'Dropout(p=%.2f)' % self.p
    


# %% test
@main
def test(*args):
    seq = Sequential(
        a := Linear(2, 5), 'tanh', Dropout(0.02),
        b := Linear(5), Dropout(0.02), 'tanh',
        c := Linear(1)
    )
    seq.setup()

    a(np.ones([10, 5]))
    
    return a

    # sm = SoftMax()
    # sm(rand.rand(10, 3))
    # error = rand.rand(10, 3)
    # for y, e, be in zip(sm.output, error, sm.backward(error)):
    #     d1 = np.diag(y) - np.outer(y, y)
    #     assert (d1 @ e == be).all()


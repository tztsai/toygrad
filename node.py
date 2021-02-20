from function import Function
from param import Parameter
from utils import *
from devtools import *


NODES = {}


def getnode(val):
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


class Node(Function, metaclass=makemeta(getnode)):
    """Base class of a computational node."""

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
        self.fan_out = fan_out
        self.fan_in = fan_in
        self.output = None
        self.input = None
        self.error = None
        self.ascendants = []
        self.descendants = []
        self.training = True
        
        self._fo = None  # private fan-out, always a list
        self._fi = None  # private fan-in, always a list
        self._forward = super().__getattribute__("forward")
        self._backward = super().__getattribute__("backward")
        
        self._has_setup = False
        self._block = False  # block forward pass to prevent looping
        
        self.id = hex(id(self))[-3:]
        while self.id in NODES:
            self.id = '0' + self.id
        NODES[self.id] = self
        
    def connect(self, *nodes):
        """Connect nodes as a descendant."""
        for node in nodes:
            node = Node(node)
            self.descendants.append(node)
            node.ascendants.append(self)
        return node  # convenient for sequential connection
            
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
    def setup(self, fan_in=None):
        """Initialize the parameters if the node has any."""
        if self._has_setup:
            dbg(f'{self} has been setup')
            return
            
        # check fan-in
        asc_fo = [node._fo[node.descendants.index(self)]
                  for node in self.ascendants]
        
        if fan_in is None:
            fan_in = self.fan_in
            
        if fan_in is None:  # set to fan-out of ascendants by default
            if not self.ascendants:
                raise AssertionError(f'input shape unspecified in {self}')
            fan_in = asc_fo
        elif self.ascendants:
            fan_in = expand(fan_in)
            assert len(fan_in) == len(asc_fo), f'input shape mismatch in {self}'
            for k, (no, ni) in enumerate(zip(asc_fo, fan_in)):
                if no is None:
                    raise AssertionError('output shape unspecified '
                                         f'in {self.ascendants[k]}')
                assert no == ni, f'input shape mismatch in {self}'
        
        self._fi = fan_in
        self.fan_in = squeeze(fan_in)
        
        # check fan-out
        fan_out = expand(self.fan_out)
        
        if fan_out is None:
            fan_out = fan_in  # the same as fan-in by default

        if self.descendants:
            assert len(fan_out) == len(self.descendants), \
                f'output shape mismatch in {self}'

        self._fo = fan_out
        self.fan_out = squeeze(fan_out)
        
        self._has_setup = True
        dbg('Setup %s', self)
        
        for node in self.descendants:
            node.setup()
        
    def _match_pass(self, nodes, data):
        if not nodes:  # no nodes to pass
            return ()
        
        if len(nodes) == 1:
            data = data,
            
        if type(data) is not tuple or len(data) != len(nodes):
            raise AssertionError("data shape mismatch when "
                                 f"passing through {self}")
        
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
        if self._block: return
        
        # call the inner forward method
        output = self._forward(*input)
        
        # record input and output
        input = squeeze(input)
        self.input, self.output = input, output
        
        # block to prevent infinite loop of forward pass
        self._block = True
        
        # pass forward the output
        for node, out in self._match_pass(self.descendants, output):
            node.forward(out)
            
        self._block = False
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
        self.error = error

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
        return f'{type(self).__name__}({self.id})'

    def __repr__(self):
        nodetype = type(self).__name__
        return nodetype + f'({self.fan_in}, {self.fan_out})'


class Network(Node):
    
    def __init__(self, nodenet: list, entry=None, exit=None):
        """Accepts a list of node connections and wrap it into a single node."""
        self.entry = entry
        self.exit = exit
        
        self.net = self.buildnet(nodenet)
        
        self.ascendants = self.entry.ascendants
        self.descendants = self.exit.descendants
        
        super().__init__(self.exit.fan_out, self.entry.fan_in)
            
    def buildnet(self, nodenet):
        net = {}
        entries, exits = [], []
        
        for node, desc in enumerate(nodenet):
            if is_list(desc):
                node.connect(*desc)
                desc = list(desc)
            else:
                node.connect(desc)        
                desc = [desc]
                
            net[node] = desc
            
            if self.exit is None:
                for d in desc:
                    if d not in net:
                        exits.append(d)
                    
        if self.entry is None:
            for node in net:
                if not any(node in desc for desc in net.values()):
                    entries.append(node)
                    
        assert len(entries) == 1, "cannot find the entry of %s" % self
        assert len(exits) == 1, "cannot find the exit of %s" % self
        self.entry = entries[0]
        self.exits = exits[0]
        
        return net
            
    def setup(self, fan_in=None):
        self.entry.setup(fan_in)
        
    def forward(self, *input):
        self.entry.forward(*input)
        return self.exit.output
    
    def backward(self, *error):
        self.exit.backward(*error)
        return self.entry.error
    
    def __repr__(self):
        # node_list = ', '.join(map(repr, self.nodes))
        return f'Network({self.net})'
        

class Linear(Node):
    """Linear transformation (actually affine transformation if bias is nonzero)."""

    def __init__(self, fan_out, fan_in=None, *, with_bias=True):
        """
        Args:
            with_bias: whether the Linear transformation has a constant bias
        """
        super().__init__(fan_out, fan_in)
        self.with_bias = with_bias

    def setup(self, fan_in=None):
        super().setup(fan_in)
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
    
    scan_step = 2
    
    def __init__(self, num_kernels, fan_in=None, kernel_width=2):
        super().__init__(fan_in=fan_in)
        self.k = kernel_width
        
    def setup(self, fan_in=None):
        super().setup(fan_in=fan_in)
        assert type(self.fan_in) is tuple and len(self.fan_in) == 2, \
            f'input dimension of {self} must be 2D'
            
        self._grid = np.array(self._make_grid())
        self.fan_out = (self.k, *self._grid.shape)
        
        self.kernels = Parameter(size=[self.num_kernels, 2*self.k + 1, 2*self.k + 1])

    def convolve(self, x, kernel):
        assert dim(x) == 2, 'data dimension should be 2D'
        k = self.k
        return [[np.sum(kernel * x[i-k:i+k+1, j-k:j+k+1]) for i, j in r]
                for r in self._grid]
        
    def forward(self, input):
        return np.array([[self.convolve(im, knl) for knl in self.kernels]
                         for im in input])
        
    def _make_grid(self):
        h, w = self.fan_in
        return [[(i, j) for j in range(self.k, w - self.k, self.scan_step)]
                for i in range(self.k, h - self.k, self.scan_step)]
        
        
class Tanh(Node):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, error):
        return error * (1 - self.output**2)


class Logistic(Node):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, error):
        return error * self.output * (1 - self.output)


class ReLU(Node):
    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, error):
        return error * (self.output > 0)


class SoftMax(Node):
    def forward(self, x):
        ex = np.exp(x - np.max(x, axis=1, keepdims=True))
        return ex / np.sum(ex, axis=1, keepdims=True)

    def backward(self, error):
        # TODO: not correct?
        dp = np.sum(error * self.output, axis=1, keepdims=True)
        return (error - dp) * self.output


class Dropout(Node):
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


# %% test
if __name__ == '__main__':
    a = Linear(5)
    b = Linear(8)
    a.connect(b)
    a.setup(3)
    a(np.ones([10, 3]))

    sigma = Node('tanh')
    print(sigma(2))

    relu = ReLU()
    relu(np.random.rand(10, 3) - 0.5)
    print(relu.backward(np.random.rand(10, 3)))

    sm = SoftMax()
    sm(np.random.rand(10, 3))
    error = np.random.rand(10, 3)
    for y, e, be in zip(sm.output, error, sm.backward(error)):
        d1 = np.diag(y) - np.outer(y, y)
        assert (d1 @ e == be).all()

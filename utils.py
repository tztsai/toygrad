import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict
from copy import deepcopy
import tqdm


def normal(size, mu=0, sigma=1):
    data = []
    for m, s in np.broadcast(mu, sigma):
        data.append(np.random.normal(loc=m, scale=s, size=size))
    return np.array(data).T


def join_classes(*classes, labels=None):
    """
    Join several classes of points into a single dataset, keeping their labels.

    Args:
        - classes: each class is an array, the columns of which are data points

    Returns:
        a dataset and a vector of labels
    """
    if labels is None:
        labels = range(len(classes))
    x = np.vstack(classes)
    y = np.hstack([[l] * len(c) for c, l in zip(classes, labels)]).reshape(-1, 1)
    data = np.hstack([y, x])
    np.random.shuffle(data)
    return data[:, 1:], data[:, 0].astype(np.int)


def init_weight(*shape):
    """Initialize a weight matrix using Xavier initialization."""
    sigma = 1 / np.sqrt(shape[0])
    return np.random.normal(scale=sigma, size=shape)


def plot_dataset(points, labels, ax=plt):
    classes = defaultdict(list)
    
    for label, point in zip(labels, points):
        classes[label].append(point)
        
    for points in classes.values():
        points = np.array(points)
        ax.plot(points[:,0], points[:,1], 'o')


def plot_history(values, *args, ax=None, title=None, xlabel='Epoch',
                 ylabel='', show=False, **kwds):
    epochs = np.arange(len(values)) + 1

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(epochs, values, *args, **kwds)
    ax.set_title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if show: plt.show()
    

def pbar(iterable, **kwds):
    """A process bar"""
    return tqdm.tqdm(iterable, bar_format='\t{l_bar}{bar:20}{r_bar}', **kwds)


def onehot(x, k):
    m = np.full((len(x), k), -1, dtype=np.int)
    for i, j in enumerate(x):
        m[i, j] = 1
    return m


def mesh_grid(xlim, ylim, nx=100, ny=100):
    """Generate a grid of points for plotting."""
    vx = np.linspace(*xlim, num=nx)
    vy = np.linspace(*ylim, num=ny)
    grid = np.array([[(x, y) for x in vx] for y in vy])
    return grid


class BatchLoader:
    """An iterable loader that produces minibatches of data."""
    batch_size = 32

    def __init__(self, *data, batch_size=batch_size):
        self.data = data
        self.size = len(data[0])
        assert all(len(x) == self.size for x in data), \
            'different sample sizes of data'
        self.bs = batch_size
        self.steps = range(0, self.size, batch_size)
        
    def __len__(self):
        return len(self.steps)

    def __iter__(self):
        order = np.random.permutation(self.size)
        for i in self.steps:
            ids = order[i : i + self.bs]
            yield [a[ids] for a in self.data]


class Animation:
    """Perceptron Training Animation"""
    time_interval = 5e-3

    def __init__(self, x, y, show_data=True):
        """
        Initialize the animation with the training data.

        Args:
            X: the inputs
            Y: the labels
        """
        assert x.shape[1] == 2
        if len(y.shape) > 1:
            assert y.shape[1] == 1
            y = y.reshape(-1)

        self.fig, self.ax = plt.subplots()

        # plot the training data
        if show_data:
            plot_dataset(x, y, self.ax)
            self.ax.autoscale()
            self.xlim = self.ax.get_xlim()
            self.ylim = self.ax.get_ylim()
        else:
            self.xlim, self.ylim = zip(np.min(x, axis=0), np.max(x, axis=0))

        self.im = self.ax.imshow([[0]], origin='lower',
                                 interpolation='bilinear',
                                 extent=[*self.xlim, *self.ylim])

    def update(self, data):
        self.im.set_clim(np.min(data), np.max(data))
        self.im.set_data(data)

        plt.ion()   # turn on interactive mode
        self.fig.canvas.draw()
        plt.pause(self.time_interval)
        plt.ioff()  # turn off interactive mode


class AnimStep:
    def __init__(self, x, y, show_data=True, binary=False, grid_size=(200, 200)):
        self.anim = Animation(x, y, show_data)
        self.binary = binary
        self._grid = mesh_grid(self.anim.xlim, self.anim.ylim, *grid_size)
        
    def __call__(self, nn):
        data = np.array([nn.forward(row).reshape(-1) for row in self._grid])
        if self.binary: data = data > 0
        self.anim.update(data)

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from abc import ABC as baseclass, abstractmethod
import numbers
import tqdm

INFO = 2
DEBUG = 1
DISPLAY_LEVEL = INFO
LOG_LEVEL = INFO


def setloglevel(level: str):
    if type(level) is str:
        d = {'info': INFO, 'debug': DEBUG}
        level = d[level.lower()]
    else:
        raise TypeError('log level is not a string')
    global LOG_LEVEL
    LOG_LEVEL = level


_print = print
def print(*msgs, **kwds):
    """Override the builtin print function."""
    if LOG_LEVEL == DISPLAY_LEVEL:
        _print(*msgs, **kwds)


def gaussian(size, mu=0, sigma=1):
    data = []
    for m, s in np.broadcast(mu, sigma):
        data.append(np.random.normal(loc=m, scale=s, size=size))
    return np.array(data).T


def join_classes(*classes, labels=None):
    """Join several classes of points into a single dataset, keeping their labels.

    Args:
        classes: each class is a 2D array whose columns are data points

    Returns:
        A dataset and a vector of labels.
    """
    if labels is None:
        labels = range(len(classes))
    x = np.vstack(classes)
    y = np.hstack([[l] * len(c) for c, l in zip(classes, labels)]).reshape(-1, 1)
    data = np.hstack([y, x])
    np.random.shuffle(data)
    return data[:, 1:], data[:, 0].astype(np.int)


def plot_dataset(points, labels, ax=plt, **kwds):
    classes = defaultdict(list)
    
    for label, point in zip(labels, points):
        classes[label].append(point)
        
    for points in classes.values():
        points = np.array(points)
        ax.plot(points[:,0], points[:,1], 'o', **kwds)


def plot_history(history, *args, title=None, **kwds):
    fig, ax = plt.subplots()
    
    for name, values in history.items():
        epochs = np.arange(len(values)) + 1
        ax.plot(epochs, values, *args, label=name, **kwds)

    ax.set_title(title)
    ax.legend()
    
    
def assure_2D(array):
    dim = len(np.shape(array))
    if dim == 1:
        return np.expand_dims(array, axis=1)
    elif dim == 2:
        return array
    else:
        return array.reshape(len(array), -1)
    
    
def pbar(iterable, **kwds):
    """A process bar."""
    if LOG_LEVEL < DISPLAY_LEVEL: return iterable
    return tqdm.tqdm(iterable, bar_format='\t{l_bar}{bar:20}{r_bar}', **kwds)


def onehot(x, k, *, cold=0, hot=1):
    m = np.full((len(x), k), cold, dtype=np.int)
    for i, j in enumerate(x):
        m[i, j] = hot
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
    """Perceptron Training Animation."""
    time_interval = 5e-3

    def __init__(self, x, y, show_data=True):
        """Initialize the animation with the training data.

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


def train_anim(x, y, show_data=True, binary=False, div_pt=0, grid_size=(200, 200)):
    """Enables animation during the neural network training.
    
    Args:
        x, y: the training data
        show_data: whether to display the training data in the animation
        binary: whether to discretize the output into 0 and 1
        div_pt: the number that divides the output values into binary classes,
            ie. an output belongs to one class if it > div_pt, otherwise the other class
        grid_size: a 2-tuple specifying the resolution of the animation
        
    Returns:
        A callback of NN.fit.
    """
    anim = Animation(x, y, show_data)
    grid = mesh_grid(anim.xlim, anim.ylim, *grid_size)
    
    def callback(nn):
        data = np.array([nn(row).squeeze() for row in grid])
        if binary: data = data > div_pt
        anim.update(data)

    return callback

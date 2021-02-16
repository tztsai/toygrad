import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Union, Optional
from abc import ABC as baseclass, abstractmethod
import numbers
import tqdm

INFO = 2
DEBUG = 1
DISPLAY_LEVEL = INFO
LOG_LEVEL = INFO

SEED = 1
np.random.seed(SEED)


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
        
        
def bernoulli(size, p=0.5):
    if len(np.shape(size)) == 0:
        size = [size]
    return (np.random.rand)(*size) < p


def gaussian(size, mu=0, sigma=1):
    data = []
    for m, s in np.broadcast(mu, sigma):
        data.append(np.random.normal(loc=m, scale=s, size=size))
    return np.array(data).T


def sign(x, eps=1e-15):
    return (x >= -eps) * 2 - 1


def join_classes(*classes, labels=None):
    """Join several classes of points into a single dataset, keeping their labels.

    Args:
        classes: each class is a 2D array whose rows are data points

    Returns:
        A dataset and a vector of labels.
    """
    if labels is None:
        labels = range(len(classes))
    x = np.vstack(classes)
    y = np.hstack([[l] * len(c) for c, l in zip(classes, labels)])[:, None]
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


def reshape2D(x, batch=True):
    n = len(x) if batch else 1
    return np.reshape(x, [n, -1])


def discretize(x, splits):
    """Discretize the data with the splitting points."""
    for i, p in enumerate(splits):
        l = splits[i - 1] if i else -np.inf
        ids = (l < x) & (x <= p)
        x[ids] = i
        if i == len(splits) - 1:
            x[x > p] = i + 1
    return x
    

class Default:
    """Change to the default value if it is set to None,
       used as a class attribute."""
    
    def __init__(self, default):
        self.default = self.value = default
        
    def __set__(self, obj, value):
        if value is not None:
            self.value = value
        else:
            self.value = self.default
            
    def __get__(self, obj, type=None):
        return self.value


class BatchLoader:
    """An iterable loader that produces minibatches of data."""
    batch_size = Default(16)

    def __init__(self, *data, batch_size=None):
        self.data = data
        self.size = len(data[0])
        assert all(len(x) == self.size for x in data), \
            'different sizes of data'
            
        self.batch_size = batch_size
        self.steps = range(0, self.size, self.batch_size)
        
    def __len__(self):
        return len(self.steps)

    def __iter__(self):
        order = np.random.permutation(self.size)
        for i in self.steps:
            ids = order[i : i + self.batch_size]
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


def train_anim(x, y, show_data=True, splits=[0], grid_size=(200, 200)):
    """Enables animation during the neural network training.
    
    Args:
        x, y: the training data
        show_data: whether to display the training data in the animation
        splits: a list of points to discretize the output
        grid_size: a 2-tuple specifying the resolution of the animation
        
    Returns:
        A callback of NN.fit.
    """
    anim = Animation(x, y, show_data)
    grid = mesh_grid(anim.xlim, anim.ylim, *grid_size)
    
    def callback(nn):
        data = np.array([nn(row).squeeze() for row in grid])
        data = discretize(data, splits)
        anim.update(data)

    return callback

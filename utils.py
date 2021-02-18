import numpy as np
import matplotlib.pyplot as plt
import numbers
import tqdm
from devtools import *


SEED = 1
np.random.seed(SEED)

        
def bernoulli(size, p=0.5):
    if len(np.shape(size)) == 0:
        size = [size]
    return (np.random.rand)(*size) < p


def gaussian(size, mu=0, sigma=1):
    data = []
    for m, s in np.broadcast(mu, sigma):
        data.append(np.random.normal(loc=m, scale=s, size=size))
    return np.array(data).T


def dim(x):
    return len(np.shape(x))


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
    labels = labels.squeeze()
    assert len(np.shape(labels)) == 1
    
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
    

def onehot(x, k, *, cold=0, hot=1):
    m = np.full((len(x), k), cold, dtype=np.int)
    for i, j in enumerate(x):
        m[i, j] = hot
    return m


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
    
        
@none_for_default
class BatchLoader:
    """An iterable loader that produces minibatches of data."""
    batch_size = 16

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
    """Animates training."""
    time_interval = 2e-3
    
    def __init__(self):
        self.fig, self.ax = plt.subplots()
    
    def step(self):
        # plt.ion()   # turn on interactive mode
        self.fig.canvas.draw()
        plt.pause(self.time_interval)
        # plt.ioff()  # turn off interactive mode
        
    @abstractmethod
    def update(self, data):
        raise NotImplementedError
        
        
class ImgAnim(Animation):
    """Animates images during training."""
    imsize = 100, 100
    
    def __init__(self, x, y, show_data=True):
        """Initialize the animation with the training data.

        Args:
            X: the inputs
            Y: the labels
        """
        assert x.shape[1] == 2
        assert y.shape[1] == 1
        assert len(x) == len(y)
        
        super().__init__()

        # plot the training data
        data_plt = plot_dataset(x, y, self.ax)
        self.ax.autoscale()
        self.xlim = self.ax.get_xlim()
        self.ylim = self.ax.get_ylim()
        
        if not show_data:
            data_plt.remove()

        self.im = self.ax.imshow(np.zeros(self.imsize),
                                 origin='lower',
                                 interpolation='bilinear',
                                 extent=[*self.xlim, *self.ylim])
        
    def update(self, data):
        self.im.set_clim(np.min(data), np.max(data))
        self.im.set_data(data)
        self.step()

class CurveAnim(Animation):
    """Animates performance curves during training."""
    
    def __init__(self, labels=['loss']):
        super().__init__()
        self.ax.set_xlabel('epoch')
        
        self.plots = []
        for label in labels:
            self.plots.extend(self.ax.plot([], label=label))
            
    def update(self, *curves):
        assert len(curves) == len(self.plots), \
            'incorrect number of curves'
            
        for curve, plot in zip(curves, self.plots):
            plot.set_xdata(np.arange(len(curve)))
            plot.set_ydata(curve)
            
        self.step()


def mesh_grid(xlim, ylim, size=(100, 100)):
    """Generate a grid of points for plotting."""
    vx = np.linspace(*xlim, num=size[0])
    vy = np.linspace(*ylim, num=size[1])
    grid = np.array([[(x, y) for x in vx] for y in vy])
    return grid


def pred_anim(x, y, show_data=True, splits=[]):
    """Animates the predictions during NN training.
    
    Args:
        x, y: the training data
        show_data: whether to display the training data in the animation
        splits: a list of points to discretize the output
        
    Returns:
        A callback of NN.fit.
    """
    anim = ImgAnim(x, y, show_data)
    grid = mesh_grid(anim.xlim, anim.ylim, ImgAnim.imsize)
    
    def callback(env):
        model = env['model']
        data = np.array([model(row).squeeze() for row in grid])
        data = discretize(data, splits)
        anim.update(data)

    return callback


def hist_anim():
    """Animates the training history of the NN."""
    anim = None
    
    def callback(env):
        nonlocal anim
        history = env['history']
        
        if anim is None:
            anim = CurveAnim(history.keys())
            
        anim.update(*history.values())
        
    return callback

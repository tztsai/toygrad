import functools
import inspect
import random
import itertools
import numpy as np
# import matplotlib.pyplot as plt
from collections import defaultdict
from .dev import progbar, setloglevel, tempset, Profile


def onehot(x, k, *, cold=0, hot=1):
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    m = np.full((*x.shape, k), cold, dtype=np.float)
    for idx in itertools.product(*map(range, x.shape)):
        m[idx][x[idx]] = hot
    return m

def standardize(x_tr, *x_ts, eps=1e-5):
    m = x_tr.mean(axis=0)
    var = x_tr.var(axis=0)
    xs = [(x - m) / np.sqrt(var + eps) for x in (x_tr, *x_ts)]
    return xs[0] if not x_ts else xs

def train_val_split(inputs, labels, ratio=0.8):
    N = len(inputs)
    idx = random.sample(range(N), int(ratio*N))
    itr = np.zeros(N, dtype=bool)
    itr[idx] = True
    return (inputs[itr], labels[itr]), (inputs[~itr], labels[~itr])

def accuracy(probabs, labels):
    preds = np.asarray(np.argmax(probabs, axis=1))
    if np.ndim(labels) == 2:  # one-hot encoded
        labels = np.argmax(labels, axis=1)
    return (preds == labels).mean() * 100


class BatchLoader:
    """An iterable loader that produces minibatches of data."""
    bs = 16
    randperm = True
    preprocess = None

    def __init__(self, *data, bs=None):
        self.data = data
        self.size = len(data[0])
        assert all(len(x) == self.size for x in data), 'different sizes of data'
        if bs: self.bs = bs
        self.steps = range(0, self.size, self.bs)

    def __len__(self):
        return len(self.steps)

    def __iter__(self):
        if self.randperm:
            order = np.random.permutation(self.size)
        else:
            order = range(self.size)
        for i in self.steps:
            ids = order[i:i+self.bs]
            yield [a[ids] if self.preprocess is None else
                   self.preprocess(a[ids]) for a in self.data]


def plot_history(history, *args, title=None, xlabel='epoch', **kwds):
    fig, ax = plt.subplots()
    for name, values in history.items():
        epochs = np.arange(len(values)) + 1
        ax.plot(epochs, values, *args, label=name, **kwds)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()

def setparnames(**bindings):
    if not bindings:
        bindings = inspect.stack()[1].frame.f_locals
    for name, par in bindings.items():
        if type(par).__name__ == 'Param':
            par.name = name

def makegif(frames, filename='_tmp_.gif', fps=60):
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_gif(filename)
    
# def discretize(x, splits):
#     """Discretize the data with the splitting points."""
#     for i, p in enumerate(splits):
#         l = splits[i - 1] if i else -np.inf
#         ids = (l < x) & (x <= p)
#         x[ids] = i
#         if i == len(splits) - 1:
#             x[x > p] = i + 1
#     return x


# class Animation:
#     """Animates training."""
#     time_interval = 2e-3
    
#     def __init__(self):
#         self.fig, self.ax = plt.subplots()
    
#     def step(self):
#         # plt.ion()   # turn on interactive mode
#         self.fig.canvas.draw()
#         plt.pause(self.time_interval)
#         # plt.ioff()  # turn off interactive mode
        
#     @abstractmethod
#     def update(self, data):
#         raise NotImplementedError
        
        
# class ImgAnim(Animation):
#     """Animates images during training."""
#     imsize = 100, 100
    
#     def __init__(self, x, y, show_data=True):
#         """Initialize the animation with the training data.

#         Args:
#             X: the inputs
#             Y: the labels
#         """
#         assert x.shape[1] == 2
#         assert y.shape[1] == 1
#         assert len(x) == len(y)
        
#         super().__init__()

#         # plot the training data
#         data_plt = plot_dataset(x, y, self.ax)
#         self.ax.autoscale()
#         self.xlim = self.ax.get_xlim()
#         self.ylim = self.ax.get_ylim()
        
#         if not show_data:
#             data_plt.remove()

#         self.im = self.ax.imshow(np.zeros(self.imsize),
#                                  origin='lower',
#                                  interpolation='bilinear',
#                                  extent=[*self.xlim, *self.ylim])
        
#     def update(self, data):
#         self.im.set_clim(np.min(data), np.max(data))
#         self.im.set_data(data)
#         self.step()

# class CurveAnim(Animation):
#     """Animates performance curves during training."""
    
#     def __init__(self, labels=['loss']):
#         super().__init__()
#         self.ax.set_xlabel('epoch')
        
#         self.plots = []
#         for label in labels:
#             self.plots.extend(self.ax.plot([], label=label))
            
#     def update(self, *curves):
#         assert len(curves) == len(self.plots), \
#             'incorrect number of curves'
            
#         for curve, plot in zip(curves, self.plots):
#             plot.set_xdata(np.arange(len(curve)))
#             plot.set_ydata(curve)
            
#         self.step()


# def mesh_grid(xlim, ylim, size=(100, 100)):
#     """Generate a grid of points for plotting."""
#     vx = np.linspace(*xlim, num=size[0])
#     vy = np.linspace(*ylim, num=size[1])
#     grid = np.array([[(x, y) for x in vx] for y in vy])
#     return grid


# def anim_train(x, y, show_data=True, splits=[]):
#     """Animates the NN training.
    
#     Args:
#         x, y: the training data
#         show_data: whether to display the training data in the animation
#         splits: a list of points to discretize the output
        
#     Returns:
#         A callback of NN.fit.
#     """
#     anim = ImgAnim(x, y, show_data)
#     grid = mesh_grid(anim.xlim, anim.ylim, ImgAnim.imsize)
    
#     def callback(env):
#         model = env['model']
#         data = np.array([model(row).squeeze() for row in grid])
#         data = discretize(data, splits)
#         anim.update(data)

#     return callback


# def hist_anim():
#     """Animates the training history of the NN."""
#     anim = None
#     def callback(env):
#         nonlocal anim
#         history = env['history']
#         if anim is None:
#             anim = CurveAnim(history.keys())
#         anim.update(*history.values())
#     return callback

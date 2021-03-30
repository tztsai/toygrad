import functools
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def dim(x):
    return len(np.shape(x))


def onehot(x, k, *, cold=0, hot=1):
    m = np.full((len(x), k), cold, dtype=np.int)
    for i, j in enumerate(x):
        m[i, j] = hot
    return m


def computation_tree(param):
    def dfs(y, visited={None}):
        try: ctx = y._ctx
        except: return
        if ctx in visited: return
        visited.add(ctx)
        tree = [y, ctx, [dfs(x, visited) for x in ctx.parents]]
    return dfs(param)


def print_comptree(param):
    # import pprint
    # ps = pprint.pformat(param)
    
    ps = str(param)



@DefaultNone
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
            ids = order[i: i + self.batch_size]
            yield [a[ids] for a in self.data]

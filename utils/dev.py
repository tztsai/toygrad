import os, time, sys
import logging
import atexit
import copy
import inspect
import itertools
import numpy as np
from contextlib import contextmanager
from logging import DEBUG, INFO, WARN, ERROR
from tqdm import tqdm
from functools import lru_cache, wraps, partial
from contextlib import contextmanager
from copy import deepcopy
from collections import defaultdict, namedtuple, OrderedDict
from typing import Union, Optional, List, Tuple
from abc import ABC, ABCMeta, abstractmethod, abstractproperty


class ProfileOp:
    debug_counts, debug_times = defaultdict(int), defaultdict(float)

    @classmethod
    def print_debug_exit(cls):
        for name, _ in sorted(cls.debug_times.items(), key=lambda x: -x[1]):
            dbg(f"{name:>20} : {cls.debug_counts[name]:>6} "
                f"{cls.debug_times[name]:>10.2f} ms")
    
    def __init__(self, name, x, backward=False):
        self.name, self.x = f"back_{name}" if backward else name, x

    def __enter__(self):
        self.st = time.time()

    def __exit__(self, *junk):
        et = (time.time()-self.st)*1000.
        self.debug_counts[self.name] += 1
        self.debug_times[self.name] += et
        dbg(f"{self.name:>20} : {et:>7.2f} ms {[np.shape(y) for y in self.x]}")

atexit.register(ProfileOp.print_debug_exit)


class LogFormatter(logging.Formatter):

    formats = {
        # DEBUG: "%(module)s.%(funcName)s, L%(lineno)s:\n  %(msg)s",
        DEBUG: "DEBUG: %(msg)s",
        INFO:  "%(msg)s",
        WARN:  "WARNING: %(msg)s",
        ERROR: "ERROR: %(msg)s"
    }
    
    def format(self, record):
        dct = record.__dict__
        fmt = LogFormatter.formats.get(record.levelno, self._fmt)
        record.msg = record.msg % record.args
        return fmt % dct

logLevel = logging.INFO

logger = logging.getLogger(__name__)
logger.setLevel(logLevel)
logFormat = LogFormatter()
logHandler = logging.StreamHandler()
logHandler.setLevel(logging.DEBUG)
logHandler.setFormatter(logFormat)
logger.addHandler(logHandler)

dbg = logger.debug
info = logger.info
warn = logger.warning

    
def setloglevel(level):
    logger.setLevel(level)

def ensure_list(a):
    return [a] if type(a) not in [list, tuple] else list(a)

def pbar(iterable, unit=' batches', **kwds):
    """A process bar."""
    if logger.level > logLevel: return iterable
    return tqdm(iterable, bar_format='\t{l_bar}{bar:24}{r_bar}', unit=unit, **kwds)
    
def signature_str(*args, **kwds):
    kwdstrs = [f'{k}={repr(v)}' for k, v in kwds.items()]
    return f"({', '.join([*map(repr, args), *kwdstrs])})"

def array_at_first(args):
    return args and isinstance(args[0], np.ndarray)

def bind_pars(f, *args, **kwds):
    s = inspect.signature(f)
    bound_pars = s.bind(*args, **kwds)
    bound_pars.apply_defaults()
    return bound_pars.arguments

def backward_stack():
    stk = inspect.stack()
    return [f.frame.f_locals['ctx'] for f in stk if f.function == '_backward']

# def copyclass(cls):
#     return type(cls)(cls.__name__, cls.__bases__, dict(cls.__dict__))

@contextmanager
def set_temporarily(obj, attr, val):
    cur_val = getattr(obj, attr)
    setattr(obj, attr, val)
    yield
    setattr(obj, attr, cur_val)
    
def DefaultNone(cls):
    """A class decorator that changes the `__getattribute__` method so that for
       instances of the decorated class, if any of its instance attribute is None and
       the corresponding class attribute exists, then returns the class attribute instead.
    """
    # keep the original __getattribute__ method
    getattribute = cls.__getattribute__
    def get(self, name):
        value = getattribute(self, name)
        if value is None and hasattr(type(self), name):
            return getattr(type(self), name)  # get the class attribute
        return value
    cls.__getattribute__ = get
    return cls

class Cache(dict):
    def __init__(self, size=128):
        self.size = size
        self.queue = []
        self._cnt = 0
        
    def __contains__(self, key):
        ret = super().__contains__(key)    
        if ret: self._cnt += 1
        return ret
    
    def __setitem__(self, key, value):
        if key not in self:
            self.queue.append(key)
        if len(self.queue) > self.size:
            for _ in range(5):
                self.pop(self.queue.pop(0))
        super().__setitem__(key, value)


# class NameSpace(dict):
#     def __getattribute__(self, name):
#         if super().__getattribute__('__contains__')(name):
#             return self[name]
#         else:
#             return super().__getattribute__(name)

#     def __setattr__(self, name, value):
#         self[name] = value


# def makemeta(getter):
#     """A metaclass factory that customizes class instantiation.
    
#     Args:
#         getter: a function that returns a class to be instantiated
        
#     Returns:
#         a metaclass that when called, returns an instance of the class returned by `getter`
#     """
#     class Meta(type):
#         def __call__(self, *args, **kwds):
#             cln = self.__name__
#             if type(self.__base__) is Meta:  # a subclass of the created class
#                 obj = self.__new__(self)
#                 obj.__init__(*args, **kwds)
#                 return obj  # initialize as usual
#             elif len(args) < 1:
#                 raise TypeError(f'{cln}() takes at least 1 argument')
#             elif isinstance(obj := args[0], self):
#                 if len(args) > 1 or kwds:  # return a new instance
#                     obj = self.__new__(type(args[0]))
#                     obj.__init__(*args[1:], **kwds)
#                 return obj
#             else:
#                 try:
#                     ret = getter(*args, **kwds)
#                 except TypeError as e:
#                     raise TypeError(f'{cln}() {e}')
#                 if isinstance(ret, self):
#                     return ret
#                 if type(ret) is tuple:
#                     cls, *ini_args = ret
#                 else:
#                     cls, ini_args = ret, ()
#                 assert issubclass(cls, self)
#                 obj = self.__new__(cls)
#                 obj.__init__(*ini_args)
#                 return obj
#     return Meta


# def swap_methods(*args):
#     """Swap method names of a class.
#     Args: a pair of str or list"""
#     if type(args[0]) is str:
#         swap = {args[0]: args[1]}
#     else:
#         swap = dict(zip(*args))
#     swap.update([[b, a] for a, b in swap.items()])

#     def deco(cls):
#         def __getattribute__(self, name):
#             if name in swap:
#                 name = swap[name]
#             return getattr(self, name)
#         getattr = cls.__getattribute__
#         cls.__getattribute__ = __getattribute__
#         return cls

#     return deco


# def decorator(dec):
#     @wraps(dec)
#     def wrapped(f):
#         return wraps(f)(dec(f))
#     return wrapped

# def slice_grid(h, w, sh, sw, dh=1, dw=1):
#     return [[(slice(i, i+sh), slice(j, j+sw))
#              for j in range(0, w - sw, dh)]
#             for i in range(0, h - sh, dw)]


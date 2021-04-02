import os, time, sys
import logging
import atexit
import inspect
import numpy as np
from logging import DEBUG, INFO, WARN, ERROR
from tqdm import tqdm
from functools import lru_cache, wraps
from contextlib import contextmanager
from copy import deepcopy
from collections import defaultdict, namedtuple
from typing import Union, Optional, List, Tuple
from abc import ABC, ABCMeta, abstractmethod


class ProfileOp:
    debug_counts, debug_times = defaultdict(int), defaultdict(float)

    @classmethod
    def print_debug_exit(cls):
        for name, _ in sorted(cls.debug_times.items(), key=lambda x: -x[1]):
            dbg(f"{name:>20} : {cls.debug_counts[name]:>6}",
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


# logging config
#%%
class LogFormatter(logging.Formatter):

    formats = {
        DEBUG: ("%(module)s.%(funcName)s, L%(lineno)s:\n  %(msg)s"),
        INFO:  "%(msg)s",
        WARN:  "WARNING: %(msg)s",
        ERROR: "ERROR: %(msg)s"
    }
    
    def format(self, record):
        dct = record.__dict__
        fmt = LogFormatter.formats.get(record.levelno, self._fmt)
        record.msg = record.msg % record.args
        return fmt % dct
        

logLevel = logging.DEBUG

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
    

def pbar(iterable, **kwds):
    """A process bar."""
    if logger.level > logLevel: return iterable
    return tqdm.tqdm(iterable, bar_format='\t{l_bar}{bar:20}{r_bar}', **kwds)


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


class NameSpace(dict):
    def __getattribute__(self, name):
        if super().__getattribute__('__contains__')(name):
            return self[name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        self[name] = value


def makemeta(getter):
    """A metaclass factory that customizes class instantiation.
    
    Args:
        getter: a function that returns a class to be instantiated
        
    Returns:
        a metaclass that when called, returns an instance of the class returned by `getter`
    """
    class Meta(type):
        def __call__(self, *args, **kwds):
            cln = self.__name__
            if type(self.__base__) is Meta:  # a subclass of the created class
                obj = self.__new__(self)
                obj.__init__(*args, **kwds)
                return obj  # initialize as usual
            elif len(args) < 1:
                raise TypeError(f'{cln}() takes at least 1 argument')
            elif isinstance(obj := args[0], self):
                if len(args) > 1 or kwds:  # return a new instance
                    obj = self.__new__(type(args[0]))
                    obj.__init__(*args[1:], **kwds)
                return obj
            else:
                try:
                    ret = getter(*args, **kwds)
                except TypeError as e:
                    raise TypeError(f'{cln}() {e}')
                if isinstance(ret, self):
                    return ret
                if type(ret) is tuple:
                    cls, *ini_args = ret
                else:
                    cls, ini_args = ret, ()
                assert issubclass(cls, self)
                obj = self.__new__(cls)
                obj.__init__(*ini_args)
                return obj
    return Meta


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


# @decorator
# def parse_name(f):
#     def call(*args, **kwds):
#         if len(args) != 1 or kwds or type(args[0]) is not str:
#             raise TypeError(f"only accepts a single str argument")
#         return f(args[0].lower())
#     return call


# def squeeze(x):
#     if is_seq(x):
#         if len(x) == 1:
#             return squeeze(x[0])
#         else:
#             return type(x)(squeeze(i) for i in x)
#     else:
#         return x
    
    
# def is_seq(x):
#     return isinstance(x, (list, tuple))


# def is_pair(x):
#     return isinstance(x, tuple) and len(x) == 2
    
    
# def seq_repr(obj):
#     if is_seq(obj):
#         return ', '.join(map(str, map(repr, obj)))
#     else:
#         return repr(obj)


# def slice_grid(h, w, sh, sw, dh=1, dw=1):
#     return [[(slice(i, i+sh), slice(j, j+sw))
#              for j in range(0, w - sw, dh)]
#             for i in range(0, h - sh, dw)]


# def int_or_pair(x):
#     if type(x) == int:
#         y = x
#     else:
#         x, y = x
#         assert type(x) is int and type(y) is int
#     return x, y
    

from tqdm import tqdm
from functools import lru_cache, wraps
from contextlib import contextmanager
from copy import deepcopy
from collections import defaultdict
from typing import Union, Optional, List
from abc import abstractmethod


DEBUG = 1
INFO = 2
WARNING = 3
ERROR = 4

VISIBLE_LEVEL = INFO  # the level at which to do printing
LOG_LEVEL = INFO      # the level of the logging, if not lower than VISIBLE_LEVEL, then log messages will be printed


_print = print

def print(*msgs, **kwds):
    """Override the builtin print function."""
    if VISIBLE_LEVEL <= LOG_LEVEL:
        _print(*msgs, **kwds)
        
        
def setloglevel(level: str):
    if type(level) is str:
        d = {'info': INFO, 'debug': DEBUG, 'warning': WARNING}
        try:
            level = d[level.lower()]
        except:
            raise ValueError("no such logging level")
    else:
        raise TypeError('log level is not str')
    
    global LOG_LEVEL
    LOG_LEVEL = level


def pbar(iterable, **kwds):
    """A process bar."""
    if LOG_LEVEL < VISIBLE_LEVEL: return iterable
    return tqdm.tqdm(iterable, bar_format='\t{l_bar}{bar:20}{r_bar}', **kwds)


def none_for_default(cls):
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


def make_meta(getter):
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
                obj = object.__new__(self)
                obj.__init__(*args, **kwds)
                return obj  # initialize as usual
            elif len(args) < 1:
                raise TypeError(f'{cln}() takes at least 1 argument')
            elif isinstance(args[0], self):
                if len(args) > 1 or kwds:  # return a new instance
                    obj = object.__new__(type(args[0]))
                    obj.__init__(*args[1:], **kwds)
                return obj
            else:
                try:
                    ret = getter(*args, **kwds)
                except TypeError as e:
                    raise TypeError(f'{cln}() {e}')
                if type(ret) is tuple:
                    cls, *ini_args = ret
                else:
                    cls, ini_args = ret, ()
                obj = object.__new__(cls)
                obj.__init__(*ini_args)
                return obj

    return Meta


def decorator(dec):
    @wraps(dec)
    def wrapped(f):
        return wraps(f)(dec(f))
    return wrapped


@decorator
def parse_name(f):
    def call(*args, **kwds):
        if len(args) != 1 or kwds or type(args[0]) is not str:
            raise TypeError(f"only accepts a single str argument")
        return f(args[0].lower())
    return call


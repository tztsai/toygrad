#!/usr/bin/env python
import gc
import unittest
from importer import *

def tensors_allocated():
  return sum([isinstance(x, Param) for x in gc.get_objects()])

class TestGC(unittest.TestCase):

  def test_gc(self):
    a = zeros(4,4)
    b = zeros(4,4)
    (a*b).mean().backward()
    assert(tensors_allocated() > 0)
    del a,b
    assert(tensors_allocated() == 0)
    
  def test_gc_complex(self):
    a = zeros(4,4)
    b = zeros(4,4)
    assert(tensors_allocated() == 2)
    (a*b).mean().backward()
    assert(tensors_allocated() == 4)
    del b
    assert(tensors_allocated() == 2)
    b = zeros(4,4)
    print(tensors_allocated())
    (a*b).mean().backward()
    print(tensors_allocated())
    assert(tensors_allocated() == 4)
    del b
    assert(tensors_allocated() == 2)

if __name__ == '__main__':
  unittest.main()

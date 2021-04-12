import sys, os

sys.path.extend(['.', '..'])
if 'tests' in os.listdir():
    os.chdir('tests')

from core import *
from op import *
from model import *
from utils import *

if __name__ == '__main__':
    from my_utils.utils import interact
    layers = Compose(
        Conv2D(32, 5, stride=2), ReLU,
        Conv2D(64, 5, stride=2), ReLU,
        Conv2D(128, 3, stride=2), ReLU,
        flatten, Affine(10)
    )
    layers(Param(size=[2, 3, 100, 100], kind=0))
    # interact()
import sys, os

sys.path.extend(['.', '..'])
if 'tests' in os.listdir():
    os.chdir('tests')

import node, model
from utils import *
from devtools import *
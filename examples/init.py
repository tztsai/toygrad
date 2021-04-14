import sys, os

if 'examples' in os.listdir():
    os.chdir('examples')
sys.path.extend(['..'])

from core import *
from op import *
from model import *
from utils import *

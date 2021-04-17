import sys, os

if 'examples' in os.listdir():
    os.chdir('examples')
sys.path.extend(['.', '..'])

from core import *
from func import *
from model import *
from utils import *

setloglevel('INFO')
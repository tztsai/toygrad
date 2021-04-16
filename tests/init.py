import sys, os

if 'tests' in os.listdir():
    os.chdir('tests')
sys.path.extend(['..'])

from core import *
from func import *
from model import *
from utils import *

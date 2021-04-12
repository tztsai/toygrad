import sys, os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import utils
import model
from model import *
from utils.graph import show_compgraph
from utils.dev import setloglevel

setloglevel('INFO')
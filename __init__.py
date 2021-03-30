import sys, os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

from utils.dev import setloglevel
setloglevel('INFO')

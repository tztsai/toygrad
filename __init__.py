import sys, os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

from devtools import setloglevel
setloglevel('INFO')

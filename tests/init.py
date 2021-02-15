import sys, os
sys.path.extend(['.', '..'])
if 'tests' in os.listdir():
    os.chdir('tests')
import sys
import os

cmd = f'taskset -c 0-15 python main.py --configs exp1'

os.system(cmd)
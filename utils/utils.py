from functools import cache
import os


@cache
def get_root():
    return os.path.dirname(os.path.dirname(os.path.abspath('main.py')))

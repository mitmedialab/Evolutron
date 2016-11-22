# coding=utf-8
import numpy as np


def get_args(kwargs, args):
    return {k: kwargs.pop(k) for k in args if k in kwargs}


def none2str(s):
    if s is None:
        return ''
    return str(s)


def shape(x):
    if not x:
        return None
    elif type(x) == np.ndarray:
        return x.shape
    elif type(x) == list and type(x[0]) == np.ndarray:
        return (len(x),) + x[0].shape
    elif type(x) == list:
        return len(x),


def probability(x):
    import argparse
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def count_lines(f):
    """ Counts the lines of large files by blocks.
    """

    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b:
                break
            yield b

    return sum(bl.count("\n") for bl in blocks(f))

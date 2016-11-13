# coding=utf-8
import numpy as np


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


class Handle(object):
    """ Handles names for loading and saving different models.
    """

    def __init__(self,
                 epochs=None,
                 batch_size=None,
                 filters=None,
                 filter_length=None,
                 model=None,
                 ftype=None,
                 program=None,
                 data_id=None,
                 **kwargs):
        self.epochs = epochs
        self.batch_size = batch_size
        self.filters = filters
        self.filter_size = filter_length

        self.model = model
        self.ftype = ftype
        self.program = program
        self.dataset = data_id

        self.filename = str(self).split('/')[-1]

    def __str__(self):
        return '{0}/{1}_{2}_{3}_{4}_{5}.{6}'.format(self.dataset,
                                                    self.filters,
                                                    self.filter_size,
                                                    self.epochs,
                                                    self.batch_size,
                                                    self.model,
                                                    self.ftype)

    def __repr__(self):
        return '{0}/{1}_{2}_{3}_{4}_{5}.{6}'.format(self.dataset,
                                                    self.filters,
                                                    self.filter_size,
                                                    self.epochs,
                                                    self.batch_size,
                                                    self.model,
                                                    self.ftype)

    def __add__(self, other):
        return str(self) + other

    def __radd__(self, other):
        return other + str(self)

    @classmethod
    def from_filename(cls, filename):
        try:
            basename, ftype, __ = filename.split('.')
        except ValueError:
            basename, ftype = filename.split('.')
        dataset = basename.split('/')[-2]

        info = basename.split('/')[-1]

        filters, filter_size, epochs, batch_size = map(int, info.split('_')[:4])

        model = info.split('_')[-1]

        obj = cls(epochs, batch_size, filters, filter_size, data_id=dataset, model=model, ftype=ftype)

        return obj

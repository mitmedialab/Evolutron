# coding=utf-8
import numpy as np

##############################
# -------- Functions ------- #
##############################


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


def nested_to_categorical(arr, classes=[]):
    flat = []
    for a in arr:
        flat += a

    u, idx = np.unique(np.concatenate((flat, classes)), return_inverse=True)

    categorical = np.zeros((len(arr), len(u)))

    count = 0
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            categorical[i, idx[count]] = 1
            count += 1

    return categorical


def nested_unique(arr):
    flat = []
    for a in arr:
        flat += a

    return np.unique(flat)

############################
# -------- Classes ------- #
############################


class Protein(object):
    def __init__(self, name, organism, aa_seq, nt_seq):
        self.name = name
        self.aa_seq = aa_seq
        self.nt_seq = nt_seq
        self.aa = str(self.aa_seq)
        self.nt = str(self.nt_seq)
        self.aa_num = len(aa_seq)
        if nt_seq:
            self.nt_num = len(nt_seq)
        else:
            self.nt_num = None
        self.org = organism


class ZincFinger(Protein):
    def __init__(self, name, organism, aa_seq, nt_seq, pwm):
        """

        :type rec_site: Nucleotide sequence of the recognition site
        """
        Protein.__init__(self, name, organism, aa_seq, nt_seq)
        self.pwm = pwm
        self.rec_site = np.power(np.ones_like(pwm) * 2.0, pwm) * .25

    def __str__(self):
        return '{x.name} {x.nt_num} nt {x.aa_num} aa'.format(x=self)


class Handle(object):
    """ Handles names for loading and saving different models.
    """

    def __init__(self,
                 epochs=None,
                 filters=None,
                 filter_length=None,
                 model=None,
                 ftype=None,
                 program=None,
                 data_id=None,
                 conv=None,
                 fc=None,
                 **kwargs):
        self.epochs = epochs
        self.filters = filters
        self.filter_size = filter_length

        self.model = model
        self.ftype = ftype
        self.program = program
        self.dataset = data_id

        self.n_convs = conv
        self.n_fc = fc

        self.filename = str(self).split('/')[-1]

    def __str__(self):
        return '{0}/{1}_{2}_{3}_{4}_{7}_{5}.{6}'.format(self.dataset,
                                                        self.filters,
                                                        self.filter_size,
                                                        self.epochs,
                                                        self.n_convs,
                                                        self.model,
                                                        self.ftype,
                                                        self.n_fc)

    def __repr__(self):
        return '{0}/{1}_{2}_{3}_{4}_{7}_{5}.{6}'.format(self.dataset,
                                                        self.filters,
                                                        self.filter_size,
                                                        self.epochs,
                                                        self.n_convs,
                                                        self.model,
                                                        self.ftype,
                                                        self.n_fc)

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

        filters, filter_size, epochs, conv, fc = map(eval, info.split('_')[:5])

        model = info.split('_')[-1]

        obj = cls(epochs=epochs, filters=filters, filter_length=filter_size, conv=conv, fc=fc,
                  data_id=dataset, model=model, ftype=ftype)

        return obj

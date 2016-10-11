# coding=utf-8


def none2str(s):
    if s is None:
        return ''
    return str(s)


def shape(x):
    try:
        return x.shape
    except AttributeError:
        return None

# TODO: make function that transforms a model from tf to theano and the opposite
# .model.np -> .saver & summaries


class Handle(object):
    def __init__(self, epochs=None,
                 batch_size=None,
                 filters=None,
                 filter_size=None,
                 model=None,
                 ftype=None,
                 program=None,
                 dataset=None,
                 extra=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.filters = filters
        self.filter_size = filter_size

        self.model = model
        self.ftype = ftype
        self.program = program
        self.dataset = dataset

        self.filename = str(self).split('/')[-1]

        self.extra = extra

    def __str__(self):
        return '{0}/{1}/{2}_{3}_{4}_{5}_{6}.{7}'.format(self.program,
                                                        self.dataset,
                                                        self.filters,
                                                        self.filter_size,
                                                        self.epochs,
                                                        self.batch_size,
                                                        self.model,
                                                        self.ftype)

    def __repr__(self):
        return '{0}/{1}/{2}_{3}_{4}_{5}_{6}.{7}'.format(self.program,
                                                        self.dataset,
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
        basename, ftype, _ = filename.split('.')

        program = basename.split('/')[-3]

        dataset = basename.split('/')[-2]

        info = basename.split('/')[-1]

        filters, filter_size, epochs, batch_size = map(int, info.split('_')[:4])

        model = info.split('_')[-1]

        obj = cls(epochs, batch_size, filters, filter_size, dataset=dataset, model=model, ftype=ftype, program=program)

        return obj


def probability(x):
    import argparse
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x

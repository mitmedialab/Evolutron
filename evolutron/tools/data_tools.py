# coding=utf-8
import numpy as np

from .seq_tools import aa2hot
from ..tools import io_tools as io


def data_it(dataset, block_size, multi_data=False):
    """ Iterates through a large array, yielding chunks of block_size.
    """
    size = len(dataset)

    for start_idx in range(0, size, block_size):
        excerpt = slice(start_idx, min(start_idx + block_size, size))
        if multi_data:
            yield [x[excerpt] for x in dataset]
        else:
            yield dataset[excerpt]


def pad_or_clip_seq(x, n):
    if n >= x.shape[0]:
        return np.concatenate((x[:n, :], np.zeros((n - x.shape[0], x.shape[1]))))
    else:
        return x[:n, :]


def load_dataset(infile, one_hot='x', padded=True, pad_y_data=False, nb_aa=20, min_aa=None, max_aa=None, codes=None,
                 code_key=None, **parser_options):
    """
    Loads the Evolutron formatted dataset from the input file. Automatically recognizes file format and calls
    corresponding parser.

    Args:
        infile:
        one_hot:
        padded:
        pad_y_data:
        nb_aa:
        min_aa:
        max_aa:
        codes:
        code_key:
        **parser_options:

    Returns: The dataset with the appropriate format given the options.

    """
    filename = infile
    filetype = filename.split('.')[-1]

    if filetype == 'tsv':
        x_data, y_data = io.csv_parser(filename, codes, code_key, sep='\t')
    elif filetype == 'csv':
        x_data, y_data = io.csv_parser(filename, codes, code_key, sep=',')
    elif filetype == 'fasta':
        x_data, y_data = io.fasta_parser(filename, codes, code_key)
    elif filetype == 'sec':
        x_data, y_data = io.secs_parser(filename, nb_aa=nb_aa, **parser_options)
    elif filetype == 'gz':
        x_data, y_data = io.npz_parser(filename, nb_aa=nb_aa, **parser_options)
    elif filetype == 'h5':
        x_data, y_data = io.h5_parser(filename, **parser_options)
    else:
        raise NotImplementedError('There is no parser for current file type.')

    if 'x' in one_hot:
        x_data = x_data.apply(lambda x: aa2hot(x, nb_aa)).tolist()

    if 'y' in one_hot:
        pass

    if padded:
        if not max_aa:
            max_aa = int(np.percentile([len(x) for x in x_data], 99))  # pad so that 99% of datapoints are complete
        else:
            max_aa = min(max_aa, np.max([len(x) for x in x_data]))

        x_data = np.asarray([pad_or_clip_seq(x, max_aa) for x in x_data], dtype=np.float32)

        if min_aa:
            min_aa = max(min_aa, np.max([len(x) for x in x_data]))
            x_data = np.asarray([pad_or_clip_seq(x, min_aa) for x in x_data], dtype=np.float32)

        if pad_y_data:
            try:
                y_data = np.asarray([pad_or_clip_seq(y, min_aa) for y in y_data])
                return x_data, y_data
            except:
                # TODO: catch this specific exception?
                pass

    data_size = len(x_data)
    print('Dataset size: {0}'.format(data_size))

    if filetype != 'h5' and y_data:
        try:
            assert ((len(x_data) == len(y_data)) or (len(x_data) == len(y_data[0])))
        except AssertionError:
            raise IOError('Unequal lengths for X ({0}) and y ({1})'.format(len(x_data), len(y_data[0])))

        y_data = np.asarray(y_data)

    return x_data, y_data

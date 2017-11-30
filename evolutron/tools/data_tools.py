# coding=utf-8
import numpy as np
import pandas as pd

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
        b = np.zeros((n, x.shape[1]))
        b[:x.shape[0]] = x
        return b
    else:
        return x[:n, :]


def pad_or_clip_img(x, n):
    assert x.shape[0] == x.shape[1], 'Image should be two dimensional with equal dimensions'
    if n >= x.shape[0]:
        b = np.zeros((n, n))
        b[:x.shape[0], :x.shape[1]] = x
        return b
    else:
        return x[:n, :n]


def random_aa_sequence(size):
    aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_probs = np.array([0.0825, 0.0135, 0.0545, 0.0675, 0.0385, 0.0705, 0.0225,
                         0.0595, 0.0585, 0.0965, 0.0245, 0.0405, 0.0475, 0.0395,
                         0.0555, 0.0665, 0.0535, 0.0685, 0.0105, 0.0295])
    return 'M' + ''.join(np.random.choice(aa, size=size, p=aa_probs))


def load_random_aa_seqs(n, length=None, min_length=100, max_length=1000):
    if length:
        return pd.Series([random_aa_sequence(length) for _ in range(n)])
    else:
        return pd.Series([random_aa_sequence(np.random.randint(min_length, max_length)) for _ in range(n)])


def preprocess_dataset(x_data, y_data=None, one_hot='x', padded=True, pad_y_data=False, nb_aa=20, min_aa=None,
                       max_aa=None):
    """

    Args:
        x_data (pd.Series):
        y_data (list or np.ndArray):
        one_hot (str):
        padded (bool):
        pad_y_data (bool):
        nb_aa:
        min_aa:
        max_aa:

    Returns:

    """
    if 'x' in one_hot:
        x_data = x_data.apply(lambda x: aa2hot(x, nb_aa)).tolist()
    else:
        x_data = x_data.tolist()

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

    if y_data:
        if padded and pad_y_data:
            y_data = np.asarray([pad_or_clip_seq(y, min_aa) for y in y_data])
        else:
            y_data = np.asarray(y_data)

        assert ((len(x_data) == len(y_data)) or (len(x_data) == len(y_data[0])))
        data_size = len(x_data)
        print('Dataset size: {0}'.format(data_size))
        return x_data, y_data
    else:
        data_size = len(x_data)
        print('Dataset size: {0}'.format(data_size))
        return x_data


def load_dataset(infile, codes=None, code_key=None, nb_aa=20, **parser_options):
    """
    Loads the Evolutron formatted dataset from the input file. Automatically recognizes file format and calls
    corresponding parser.

    Args:
        infile:
        codes:
        code_key:
        nb_aa:
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
    return x_data, y_data

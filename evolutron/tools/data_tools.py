# coding=utf-8
import numpy as np

from evolutron.tools import io_tools as io


def data_it(dataset, block_size):
    """ Iterates through a large array, yielding chunks of block_size.
    """
    size = len(dataset)

    for start_idx in range(0, size, block_size):
        excerpt = slice(start_idx, min(start_idx + block_size, size))
        yield dataset[excerpt]


def load_dataset(data_id, **parser_options):
    """Fetches the correct dataset from database based on data_id.
    """

    dataset = io.tab_parser(data_id, **parser_options)

    try:
        assert type(dataset) == np.ndarray or type(dataset) == list
    except AssertionError:
        if data_id == 'type2p':
            dataset, _ = io.type2p(**parser_options)
        elif data_id == 't2pneb':
            dataset = io.type2p(**parser_options)
        elif data_id == 'c2h2':
            dataset, _ = io.b1h(**parser_options)
        elif data_id == 'b1h':
            dataset = io.b1h(**parser_options)
        elif data_id == 'swissprot':
            dataset = io.fasta_parser('datasets/uniprot_sprot.fasta', **parser_options)
        elif data_id == 'cas9':
            dataset = io.fasta_parser('datasets/uniprot_cas9.fasta', **parser_options)
        elif data_id == 'random':
            dataset = io.fasta_parser('datasets/random_aa.fasta', **parser_options)
        elif data_id == 'type2':
            dataset = io.fasta_parser('datasets/uniprot_type2.fasta', **parser_options)
        elif data_id == 'm6a':
            dataset = io.m6a(**parser_options)
        else:
            print('Something went terribly wrong...')
            return 1

    if len(dataset) > 2:
        # Unsupervised Learning
        # x: observations
        x_data = dataset

        # If sequences are padded, transform data list into a numpy array for mini-batching
        if all(x.shape == x_data[0].shape for x in x_data):
            x_data = np.asarray(x_data, dtype=np.float32)

        print('Dataset size: {0}'.format(len(dataset)))

        return x_data

    elif len(dataset) == 2:
        # Supervised Learning
        # x: observations
        # y: class labels
        (x_data, y_data) = dataset
        try:
            assert (len(x_data) == len(y_data))
        except AssertionError:
            print('Unequal lengths for X ({0}) and y ({1})'.format(len(x_data), len(y_data)))
            raise IOError
        data_size = len(x_data)

        # If sequences are padded, transform data list into a numpy array for mini-batching
        try:
            x_data = np.asarray(x_data, dtype=np.float32)
            y_data = np.asarray(y_data, dtype=np.float32)
        except ValueError:
            pass

        print('Dataset size: {0}'.format(data_size))

        return x_data, y_data
    else:
        raise IOError("Dataset import fault.")

# coding=utf-8
from .io_tools import *
import pandas as pd


def data_it(dataset, block_size):
    size = len(dataset)

    for start_idx in range(0, size, block_size):
        excerpt = slice(start_idx, min(start_idx + block_size, size))
        yield dataset[excerpt]


# noinspection PyShadowingNames
def load_dataset(data_id, shuffled=False, **parser_options):
    """
    Load the dataset 
    :return: train, validation and test sets
    """

    if data_id == 'type2p':
        dataset, _ = type2p(**parser_options)
    elif data_id == 't2pneb':
        dataset = type2p(**parser_options)
    elif data_id == 'c2h2':
        dataset, _ = b1h(**parser_options)
    elif data_id == 'b1h':
        dataset = b1h(**parser_options)
    elif data_id == 'swissprot':
        dataset = fasta_parser('datasets/uniprot_sprot.fasta', **parser_options)
    elif data_id == 'cas9':
        dataset = fasta_parser('datasets/uniprot_cas9.fasta', **parser_options)
    elif data_id == 'zinc':
        dataset = fasta_parser('datasets/uniprot_zinc.fasta', **parser_options)
    elif data_id == 'homeo':
        dataset = fasta_parser('datasets/uniprot_homeo.fasta', **parser_options)
    elif data_id == 'ecoli':
        dataset = fasta_parser('datasets/uniprot_ecoli.fasta', **parser_options)
    elif data_id == 'hsapiens':
        try:
            dataframe = pd.read_hdf('datasets/uniprot_hsapiens.h5', 'table')
        except IOError:
            dataframe = tab_parser('datasets/uniprot_hsapiens.tsv')

        dataset = dataframe.x_data.tolist()
    elif data_id == 'dnabind':
        dataset = fasta_parser('datasets/uniprot_dnabind.fasta', **parser_options)
    elif data_id == 'random':
        dataset = fasta_parser('datasets/random_aa.fasta', **parser_options)
    elif data_id == 'type2':
        dataset = fasta_parser('datasets/uniprot_type2.fasta', **parser_options)
    elif data_id == 'm6a':
        dataset = m6a(**parser_options)
    else:
        print('Something went terribly wrong...')
        return 1

    if len(dataset) > 2:
        # Unsupervised Learning
        # x: observations
        x_data = dataset

        # Shuffle the data to have un-biased minibatches
        if shuffled:
            np.random.shuffle(x_data)

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

        # Shuffle the data to have un-biased minibatches
        if shuffled:
            rng_state = np.random.get_state()
            np.random.shuffle(x_data)
            np.random.set_state(rng_state)
            np.random.shuffle(y_data)

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


def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b:
            break
        yield b


def count_lines(f):
    return sum(bl.count("\n") for bl in blocks(f))

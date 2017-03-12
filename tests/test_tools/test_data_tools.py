# coding=utf-8
"""
Test of loading datasets.

"""
import pytest
import numpy as np
from evolutron.tools import load_dataset


def test_load_dataset():
    print('\nTesting padding...')

    x_data, y_data = load_dataset('random', padded=False)
    assert type(x_data) == list

    x_data, y_data = load_dataset('random', padded=True)
    assert type(x_data) == np.ndarray

    x_data, y_data = load_dataset('random', padded=True, max_aa=500)
    assert x_data.shape[1] == 500


# noinspection PyTypeChecker
def test_file_db():
    print('\nTesting database loading...')

    with pytest.raises(IOError):
        load_dataset('key_error')

    # TODO: Add more representative datasets
    for data_id in ['random', 'type2p', 'ecoli']:
        print('\nDataset: {} '.format(data_id))
        _, _ = load_dataset(data_id, padded=True, max_aa=1000)

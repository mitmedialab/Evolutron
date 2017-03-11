"""
Test of loading datasets.

Current datasets: type2p, b1h, m6a
"""
from evolutron.tools import load_dataset
from evolutron.tools.io_tools import *


def test_not_padded():
    print('\nTesting without padding')
    for shuffled in [True, False]:
        raw_data = type2p(padded=False)
        x_data, y_data = load_dataset(raw_data, shuffled=shuffled)

        assert (len(x_data) == len(y_data))

        raw_data = m6a(padded=False)
        x_data, y_data = load_dataset(raw_data, shuffled=shuffled)

        assert (len(x_data) == len(y_data))

        raw_data = b1h(padded=False)
        x_data, y_data = load_dataset(raw_data, shuffled=shuffled)

        assert (len(x_data) == len(y_data))


def test_padded():
    print('\nTesting with padding')
    for shuffled in [True, False]:
        raw_data = type2p(padded=True)
        x_data, y_data = load_dataset(raw_data, shuffled=shuffled)

        assert (len(x_data) == len(y_data))

        raw_data = m6a(padded=True)
        x_data, y_data = load_dataset(raw_data, shuffled=shuffled)

        assert (len(x_data) == len(y_data))

        raw_data = b1h(padded=True)
        x_data, y_data = load_dataset(raw_data, shuffled=shuffled)

        assert (len(x_data) == len(y_data))



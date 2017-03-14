# coding=utf-8
import pytest
import os
import pandas as pd

from evolutron.tools import io_tools as io


# noinspection PyTypeChecker
def test_fasta_parser():

    x_data, y_data = io.fasta_parser('tests/test_tools/samples/no_codes.fasta', codes=False)

    assert type(x_data) == pd.Series
    assert not y_data

    with pytest.raises(IOError):
        io.fasta_parser('tests/test_tools/samples/no_codes.fasta', codes=True)

    x_data, y_data = io.fasta_parser('tests/test_tools/samples/type2p_codes.fasta', codes=False)

    assert type(x_data) == pd.Series
    assert not y_data

    x_data, y_data = io.fasta_parser('tests/test_tools/samples/type2p_codes.fasta', codes=True, code_key='type2p')

    assert type(x_data) == pd.Series
    assert type(y_data) == list


def test_tab_parser():

    x_data, y_data = io.tab_parser('tests/test_tools/samples/sample.tsv', codes=False)

    assert type(x_data) == pd.Series
    assert not y_data

    x_data, y_data = io.tab_parser('tests/test_tools/samples/sample.tsv', codes=True, code_key='fam')

    assert type(x_data) == pd.Series
    assert type(y_data) == list

    x_data, y_data = io.tab_parser('tests/test_tools/samples/sample.tsv', codes=False)

    assert type(x_data) == pd.Series
    assert not y_data

    # Cleaning up
    os.remove('tests/test_tools/samples/sample.h5')


def test_secs_parser():
    x_data, y_data = io.secs_parser('tests/test_tools/samples/smallSecS.sec')

    assert x_data, y_data


def test_npz_parser():
    x_data, y_data = io.npz_parser('/data/datasets/cb513+profile_split1.npy.gz')

    assert x_data, y_data

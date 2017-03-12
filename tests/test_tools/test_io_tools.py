# coding=utf-8
import pytest
import os
from evolutron.tools import io_tools as io


# noinspection PyTypeChecker
def test_fasta_parser():

    x_data, y_data = io.fasta_parser('tests/test_tools/samples/no_codes.fasta', codes=False)

    assert x_data, not y_data

    with pytest.raises(IOError):
        io.fasta_parser('tests/test_tools/samples/no_codes.fasta', codes=True)

    x_data, y_data = io.fasta_parser('tests/test_tools/samples/type2p_codes.fasta', codes=False)

    assert x_data, not y_data

    x_data, y_data = io.fasta_parser('tests/test_tools/samples/type2p_codes.fasta', codes=True, code_key='type2p')

    assert x_data, y_data


def test_tab_parser():

    x_data, y_data = io.tab_parser('tests/test_tools/samples/sample.tsv', codes=False)

    x_data, y_data = io.tab_parser('tests/test_tools/samples/sample.tsv', codes=True, code_key='fam')

    x_data, y_data = io.tab_parser('tests/test_tools/samples/sample.tsv', codes=False, nb_aa=22)

    # Cleaning up
    os.remove('tests/test_tools/samples/sample.h5')


def test_secs_parser():
    # TODO: implement this
    pass


def test_npz_parser():
    # TODO: implement this
    pass

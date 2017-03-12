# coding=utf-8

from evolutron.tools.seq_tools import (aa2hot, hot2aa, secs2hot, hot2SecS_3cat, hot2SecS_8cat,
                                       ntround, nt2prob, aa2codon)


def test_converters():
    assert hot2aa(aa2hot('MKFLKIK')) == 'MKFLKIK'

    assert hot2SecS_3cat(secs2hot('HLHLEE', 3)) == 'HCHCEE'

    assert hot2SecS_8cat(secs2hot('HLHLEE', 8)) == 'HCHCEE'

    # TODO: write tests for ntround, nt2prob, aa2codon

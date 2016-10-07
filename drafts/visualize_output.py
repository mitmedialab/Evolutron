#!/usr/bin/env python

import os
import sys

import corebio
import numpy as np
from weblogolib import *

import type2restriction as t2re

from helpers.parse_type2 import num2aa


# noinspection PyShadowingNames
def main(filename):
    # Load network parameters
    with np.load(filename) as f:
        x_test = f['x_test']
        y_pred = f['y_pred']
        y_test = f['y_test']

    type2p_res_aa = {
        str(t2re.type2re[t].aa_seq) + ''.join(['M' for i in xrange(700 - len(t2re.type2re[t].aa_seq))]): t2re.type2re[
            t].name for t in t2re.type2re}

    x_test_aa = []
    for i, seq in enumerate(x_test):
        x_test_aa.append(type2p_res_aa[''.join(num2aa(seq))])

    y_pred = y_pred.reshape((-1, 6, 4))
    y_test = y_test.reshape((-1, 6, 4))

    assert (y_pred.shape == y_test.shape)

    # for k, pred in enumerate(y_pred):
    #     for i in xrange(len(pred)):
    #         y_pred[k, i] = (pred[i] - min(pred[i])) / (max(pred[i]) - min(pred[i]))
    #         y_pred[k, i] /= sum(pred[i])

    print('Started visualizations')
    foldername = 'outputs/type2p/' + filename[16:-10]
    if not os.path.isdir(foldername):
        os.mkdir(foldername)
    for i, (pred_motif, test_motif) in enumerate(zip(y_pred, y_test)):
        seq_name = x_test_aa[i]
        data = LogoData.from_counts(corebio.seq_io.unambiguous_dna_alphabet, pred_motif)
        options = LogoOptions()
        options.title = seq_name
        my_format = LogoFormat(data, options)
        my_pdf = pdf_formatter(data, my_format)
        foo = open(foldername + '/' + seq_name + '_pred.pdf', "w")
        foo.write(my_pdf)
        foo.close()

        seq_name = x_test_aa[i]
        data = LogoData.from_counts(corebio.seq_io.unambiguous_dna_alphabet, test_motif)
        options = LogoOptions()
        options.title = seq_name
        my_format = LogoFormat(data, options)
        my_pdf = pdf_formatter(data, my_format)
        foo = open(foldername + '/' + seq_name + '_test.pdf', "w")
        foo.write(my_pdf)
        foo.close()

    print('Output available')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        main(filename)
    else:
        exit(1)

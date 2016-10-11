#!/usr/bin/env python

from __future__ import print_function

import os

import numpy as np
import weblogolib as wl
from corebio.seq_io import SeqList

from evolutron.tools import hot2aa, data_it


class Motif(object):

    def __init__(self, seqs):

        self.seqs = SeqList(seqs)

        self.seqs.alphabet = wl.std_alphabets['protein']

        self.data = wl.LogoData.from_seqs(self.seqs)

        self.entropy = self.data.entropy

    def is_good(self):
        if np.max(self.data.entropy) < 2.0:
            return False
        return True


# noinspection PyShadowingNames
def motif_extraction(motif_fun, x_data, handle, depth=1, filters=None, filter_size=None):
    foldername = 'motifs/' + str(handle).split('.')[0] + '/{0}/'.format(depth)
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    if not filters:
        filters = np.squeeze(motif_fun([x_data[0]]), 0).shape[0]
    if not filter_size:
        filter_size = x_data[0].shape[1] - np.squeeze(motif_fun([x_data[0]]), 0).shape[1] + 1

    # Filter visual field
    vf = filter_size + depth * (filter_size - 1)

    max_seq_scores = []
    # Calculate the activations for each filter for each protein in data set
    for x_part in data_it(x_data, 5000):
        seq_scores = iter(np.squeeze(motif_fun([y]), 0) for y in x_part)

        # For every filter, keep max and argmax for each input protein
        max_seq_scores.append(np.asarray([np.vstack((np.max(x, 1), np.argmax(x, 1))) for x in seq_scores]))

        del seq_scores

    max_seq_scores = np.concatenate(max_seq_scores).transpose((2, 0, 1))

    # noinspection PyUnusedLocal
    matches = [[] for i in range(filters)]
    for k, filt in enumerate(max_seq_scores):
        seq_mean = np.mean(filt[:, 0])
        # seq_mean = 0
        seq_std = np.std(filt[:, 0])
        for i, seq in enumerate(filt):
            if seq[0] > seq_mean + 3 * seq_std:
                j = int(seq[1])
                if j + vf - 1 < x_data[i].shape[1]:
                    matches[k].append(hot2aa(x_data[i][:, j:j + vf]))

    del max_seq_scores

    motifs = generate_motifs(matches)
    print('Extracted {0} motifs'.format(len(motifs)))

    generate_logos(motifs, foldername)
    print("Generating Sequence Logos")

    return


def generate_motifs(matches):
    motifs = []
    for match in matches:
        if len(match) > 0:
            motif = Motif(match)
            if motif.is_good():
                motifs.append(motif)
    return motifs


def generate_logos(motifs, foldername):

    options = wl.LogoOptions()
    options.color_scheme = wl.std_color_schemes["chemistry"]

    for i, motif in enumerate(motifs):
        my_format = wl.LogoFormat(motif.data, options)
        # my_png = wl.png_print_formatter(motif.data, my_format)
        my_pdf = wl.pdf_formatter(motif.data, my_format)
        # foo = open(foldername + '/' + str(i) + ".png", "w")
        # foo.write(my_png)
        # foo.close()
        foo = open(foldername + str(i) + ".pdf", "w")
        foo.write(my_pdf)
        foo.close()
        foo = open(foldername + str(i) + ".txt", "w")
        for seq in motif.seqs:
            foo.write("%s\n" % str(seq))
        foo.close()

    return

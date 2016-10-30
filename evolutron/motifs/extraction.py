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
def motif_extraction(motif_fun, x_data, filters, filter_length, handle, depth):
    foldername = 'motifs/' + str(handle).split('.')[0] + '/{0}/'.format(depth + 1)
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    # Filter visual field
    vf = filter_length + depth * (filter_length - 1)

    # Calculate the activations for each filter for each protein in data set
    max_seq_scores = []
    for x_part in data_it(x_data, 5000):
        seq_scores = [np.squeeze(motif_fun([[y]]), 0) for y in x_part]

        # For every filter, keep max and argmax for each input protein
        max_seq_scores.append(np.asarray([np.vstack((np.max(x, 0), np.argmax(x, 0))) for x in seq_scores]))

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

    print([len(m) for m in matches])

    motifs = generate_motifs(matches)
    print('Extracted {0} motifs'.format(len([1 for m in motifs if m])))

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
            else:
                motifs.append(None)
    return motifs


def generate_logos(motifs, foldername):

    options = wl.LogoOptions()
    options.color_scheme = wl.std_color_schemes["chemistry"]

    for i, motif in enumerate(motifs):
        if motif:
            my_format = wl.LogoFormat(motif.data, options)
            # my_png = wl.png_print_formatter(motif.data, my_format)
            my_pdf = wl.pdf_formatter(motif.data, my_format)
            # foo = open(foldername + '/' + str(i) + ".png", "w")
            # foo.write(my_png)
            # foo.close()
            foo = open(foldername + str(i) + ".pdf", "w")
            foo.write(my_pdf)
            foo.close()
            # foo = open(foldername + str(i) + ".txt", "w")
            # for seq in motif.seqs:
            #     foo.write("%s\n" % str(seq))
            # foo.close()
    return

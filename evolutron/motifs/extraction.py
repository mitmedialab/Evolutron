#!/usr/bin/env python

from __future__ import print_function

import os
import shutil

import numpy as np
import weblogolib as wl
from corebio.seq_io import SeqList

from evolutron.tools import data_it, hot2aa


def make_pfm(layer_weights):
    max_scale = layer_weights.max(axis=-1).max(axis=-1)[...,
                                                        np.newaxis, np.newaxis]
    min_scale = layer_weights.min(axis=-1).min(axis=-1)[...,
                                                        np.newaxis, np.newaxis]
    return (10000 * (layer_weights - min_scale) /
            (max_scale - min_scale)).astype('uint16')


class Motif(object):
    def __init__(self, seqs):
        self.seqs = SeqList(seqs)

        self.seqs.alphabet = wl.std_alphabets['protein']

        self.data = wl.LogoData.from_seqs(self.seqs)

        self.entropy = self.data.entropy

    @property
    def is_good(self):
        if np.max(self.data.entropy) < 2.0:
            return False
        return True


# noinspection PyShadowingNames
def motif_extraction(motif_fun, x_data, filters, kernel_size, output_foldername, depth,
                     multi_input=False, filetype='txt+png'):
    foldername = os.path.join(output_foldername, str(depth + 1))

    if not os.path.exists(foldername):
        os.makedirs(foldername)
    else:
        shutil.rmtree(foldername)
        os.makedirs(foldername)

    # Filter visual field
    vf = kernel_size + depth * (kernel_size - 1)

    # Calculate the activations for each filter for each protein in data set
    max_seq_scores = []
    for x_part in data_it(x_data, 1000, multi_data=multi_input):
        if multi_input:
            seq_scores = np.squeeze(motif_fun(x_part), 0)
        else:
            seq_scores = np.squeeze(motif_fun([x_part]), 0)

        # For every filter, keep max and argmax for each input protein
        max_seq_scores.append(np.asarray([np.vstack((np.max(x, 0), np.argmax(x, 0))) for x in seq_scores]))
        del seq_scores

    max_seq_scores = np.concatenate(max_seq_scores).transpose((2, 0, 1))
    # noinspection PyUnusedLocal
    matches = [[] for i in range(filters)]
    for k, filt in enumerate(max_seq_scores):
        seq_mean = np.mean(filt[:, 0])
        seq_std = np.std(filt[:, 0])
        for i, seq in enumerate(filt):
            if seq[0] > seq_mean + 3 * seq_std:
                j = int(seq[1])
                if j + vf - 1 < x_data[i].shape[0]:
                    matches[k].append(hot2aa(x_data[i][j:j + vf, :]))

    del max_seq_scores

    motifs = generate_motifs(matches)
    print('Extracted {0} motifs'.format(len([1 for m in motifs if m])))

    generate_logos(motifs, foldername, filetype)
    print("Generating Sequence Logos")
    return


def generate_motifs(matches):
    motifs = []
    for match in matches:
        if len(match) > 0:
            motif = Motif(match)
            if motif.is_good:
                motifs.append(motif)
            else:
                motifs.append(None)
    return motifs


def generate_logos(motifs, foldername, filetype='png'):
    options = wl.LogoOptions()
    options.color_scheme = wl.std_color_schemes["chemistry"]

    for i, motif in enumerate(motifs):
        if motif:
            my_format = wl.LogoFormat(motif.data, options)
            if 'png' in filetype:
                to_write = wl.png_print_formatter(motif.data, my_format)
                foo = open(os.path.join(foldername, str(i) + '_' + str(len(motif.seqs)) + ".png"), "wb")
                foo.write(to_write)
                foo.close()
            if 'pdf' in filetype:
                to_write = wl.pdf_formatter(motif.data, my_format)
                foo = open(os.path.join(foldername, str(i) + '_' + str(len(motif.seqs)) + ".pdf"), "wb")
                foo.write(to_write)
                foo.close()
            if 'txt' in filetype:
                to_write = ''.join(["%s\n" % str(seq) for seq in motif.seqs])
                foo = open(os.path.join(foldername, str(i) + '_' + str(len(motif.seqs)) + ".txt"), "w")
                foo.write(to_write)
                foo.close()
            else:
                raise ValueError('Invalid filetype. Available options: png, pdf or txt. ')

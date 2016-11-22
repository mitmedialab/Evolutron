#!/usr/bin/env python

from __future__ import print_function

import argparse

import numpy as np
import theano
import lasagne

# Check if package is installed, else fallback to developer mode imports
try:
    import evolutron.networks.las.comet as nets
    from evolutron.tools import load_dataset, Handle, shape
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.abspath('..'))

    import evolutron.networks.las.comet as nets
    from evolutron.tools import load_dataset, Handle, shape


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
        seq_scores = iter(np.squeeze(motif_fun([[y]]), 0) for y in x_part)

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


def main(filename, data_id):
    # Load network parameters
    with np.load(filename) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    filters = param_values[0].shape[0]
    filter_size = param_values[0].shape[2]
    hidden_size = param_values[-2].shape[1]
    # TODO: make sure this works for multiple conv layers

    handle = Handle.from_filename(filename)

    if data_id == 'model':
        data_id = handle.dataset

    x_data = load_dataset(data_id, shuffled=False)

    # Load Network
    if handle.model == 'CoDER':
        net = nets.CoDER(shape(x_data), handle.batch_size, filters, filter_size)
    elif handle.model == 'CoHST':
        net = nets.CoHST(None, filters, filter_size)
    elif handle.model == 'CoHSTCoDER':
        net = nets.CoDER(None, filters, filter_size)
    elif handle.model == 'CoDERCoHST':
        net = nets.CoHST(None, filters, filter_size)
    elif handle.model == 'DeepCoDER':
        net = nets.DeepCoDER(shape(x_data), handle.batch_size, 3, hidden_size, filters, filter_size)
    else:
        if 'b1h' in handle:
            net = nets.ConvZFb1h(None, filters, filter_size)
        elif 'type2p' in handle:
            net = nets.ConvType2p(None, filters, filter_size)
        elif 'm6a' in handle:
            net = nets.ConvM6a(None, filters, filter_size, False)
        else:
            raise NotImplementedError('Model not able to be visualized at this moment.')

    conv_layers = [x for x in lasagne.layers.get_all_layers(net.network) if x.name.find('Conv') == 0]

    depth = len(conv_layers)

    for i, c in enumerate(conv_layers):
        c.W.set_value(param_values[2 * i])
        c.b.set_value(param_values[2 * i + 1])

    for i in range(0, depth):
        conv_scores = lasagne.layers.get_output(conv_layers[i])  # Changed from -1 to 0

        # Compile function that spits out the outputs of the correct convolutional layer
        motif_fun = theano.function(net.inp.values(), conv_scores)
        # Start visualizations

        motif_extraction(motif_fun, x_data, handle, i, filters, filter_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network visualization module.')
    parser.add_argument("model", help='Path to the file')

    parser.add_argument("-d", "--dataset", type=str, default='model',
                        help='Dataset on which the motifs will be generated upon. Write "model" to infer' \
                             'automatically from model.')

    args = parser.parse_args()

    kwargs = {'filename': args.model,
              'data_id': args.dataset}

    main(**kwargs)

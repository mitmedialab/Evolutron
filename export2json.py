#!/usr/bin/env python

from __future__ import print_function

import argparse
import json

import lasagne
import numpy as np
import theano
import theano.tensor as ten

import evolutron.tools.io_tools as parse
import evolutron.engine.las as nets
from evolutron.tools import load_dataset, num2aa, hot2num


# noinspection PyShadowingNames
def get_matches(motif_fun, out_fun, x_train):

    # Calculate the motif scores and output scores for each protein in data set
    motif_scores = [np.squeeze(motif_fun([x]), 0) for x in x_train]
    output_scores = [np.squeeze(out_fun([x]), 0) for x in x_train]

    max_seq_scores = np.asarray([np.vstack((np.max(x, 1), np.argmax(x, 1))).transpose() for x in motif_scores])

    scored_inputs = []

    for ind, mat in enumerate(max_seq_scores):
        seq = ''.join(num2aa(hot2num(x_train[ind])))  # TODO: here I should get also the name of the protein
        matches = np.asarray(map(lambda x: np.asarray([0, x[1]]) if x[0] < 1 else x, mat))
        scored_inputs.append([seq, map(float, matches[:, 0]), map(int, matches[:, 1])])

    return scored_inputs, np.asarray(output_scores).tolist()


def main(filename):
    # Load network parameters
    with np.load(filename) as f:
        [_, _, filters, filter_size] = f['arr_0']
        param_values = [f['arr_%d' % i] for i in range(1, len(f.files))]

    # Load Network and training data
    if filename.find('b1h') > 0:
        inputs = ten.tensor3('inputs')
        network = nets.build_network_zinc_fingers(inputs, filters, filter_size)
        raw_data = parse.b1h(padded=False)
    elif filename.find('m6a') > 0:
        inputs = ten.tensor3('inputs')
        network = nets.build_network_m6a(inputs, filters, filter_size)
        raw_data = parse.m6a(padded=False,probe='1')
    elif filename.find('type2p_pad') > 0:
        inputs = ten.tensor3('inputs')
        network = nets.build_network_type2p_padded(inputs, 700, filters, filter_size)
        raw_data = parse.type2p(padded=True)
    elif filename.find('type2p') > 0:
        inputs = ten.tensor3('inputs')
        network = nets.build_network_type2p(inputs, filters, filter_size)
        raw_data = parse.type2p(padded=False)
    else:
        raise IOError('Unrecognizable network file')
    # Set trained values to weights and biases
    x_train, = load_dataset(raw_data, shuffled=False)

    lasagne.layers.set_all_param_values(network, param_values)

    # Compile function that spits out the outputs of the convolutional layer
    if filename.find('m6a') > 0:
        scores = network.input_layer.input_layer.input_layer.get_output_for(inputs)
        w_out = param_values[-2].tolist()
        b_out = param_values[-1].tolist()
    else:
        scores = network.input_layers[0].input_layer.input_layer.input_layer.get_output_for(inputs)
        w_out = np.concatenate(param_values[2::2], axis=1).tolist()
        b_out = np.concatenate(param_values[3::2], axis=0).tolist()

    motif_fun = theano.function([inputs], scores)

    outputs = lasagne.layers.get_output(network, deterministic=True)

    out_fun = theano.function([inputs], outputs)

    scored_inputs, output_scores = get_matches(motif_fun, out_fun, x_train)

    with open(filename + '.json', 'w') as outfile:
        json.dump({"num_filters": filters,
                   "size_filter": filter_size,
                   "out_weights": [w_out],
                   "out_biases": [b_out],
                   "output_scores": output_scores,
                   "scored_inputs": scored_inputs}, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network exporting module.')
    parser.add_argument("file", help='Path to the file')
    args = parser.parse_args()

    kwargs = {'filename': args.file}

    main(**kwargs)

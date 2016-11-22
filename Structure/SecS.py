#!/usr/bin/env python
# coding=utf-8
"""
    SecS - Secondary Structure
    ------------------------------------------_
    SecStructure is an automated tool for prediction of protein secondary structure
     from it's amino acid sequence.

    (c) Massachusetts Institute of Technology

    For more information contact:
    kfirs [at] mit.edu
"""

import argparse
import time

import numpy as np

# Check if package is installed, else fallback to developer mode imports
try:
    import evolutron.networks as nets
    from evolutron.tools import load_dataset, none2str, Handle
    from evolutron.engine import DeepTrainer
    from evolutron.networks.krs.SecS import DeepCoDER
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.abspath('..'))
    import evolutron.networks as nets
    from evolutron.tools import load_dataset, none2str, Handle, shape
    from evolutron.engine import DeepTrainer
    from evolutron.networks.krs.SecS import DeepCoDER


def supervised(x_data, y_data, handle,
               epochs=10,
               batch_size=128,
               filters=8,
               filter_length=10,
               validation=.3,
               optimizer='nadam',
               rate=.01,
               conv=1,
               fc=1,
               lstm=1,
               nb_categories=8,
               dilation=1,
               model=None):

    filters = nb_categories

    # Find input shape
    if type(x_data) == np.ndarray:
        input_shape = x_data[0].shape
    elif type(x_data) == list:
        input_shape = (None, x_data[0].shape[1])
    else:
        raise TypeError('Something went wrong with the dataset type')

    if model:
        conv_net = DeepTrainer(nets.DeepCoDER.from_saved_model(model))
        print('Loaded model')
    else:
        print('Building model ...')
        net_arch = DeepCoDER.from_options(input_shape,
                                           n_conv_layers=conv,
                                           n_fc_layers=fc,
                                           use_lstm=lstm,
                                           n_filters=filters,
                                           filter_length=filter_length,
                                           nb_categories=nb_categories,
                                          dilation=dilation)
        handle.model = 'realDeepCoDER'
        conv_net = DeepTrainer(net_arch)
        conv_net.compile(optimizer=optimizer, lr=rate)

    conv_net.display_network_info()

    print('Started training at {}'.format(time.asctime()))

    conv_net.fit(x_data, y_data,
                 nb_epoch=epochs,
                 batch_size=batch_size,
                 validate=validation,
                 patience=100
                 )

    #print('Testing model ...')
    #score = conv_net.score(x_data[int(.9 * len(x_data)):], x_data[int(.9 * len(x_data)):])
    #print('Test Loss:{0:.6f}, Test Accuracy: {1:.2f}%'.format(score[0], 100 * score[1]))

    conv_net.save_train_history(handle)
    conv_net.save_model_to_file(handle)


def get_args(kwargs, args):
    return {k: kwargs.pop(k) for k in args if k in kwargs}


def main(**options):
    if 'model' in options:
        handle = Handle.from_filename(options.get('model'))
        assert handle.program == 'CoMET', 'The model file provided is for another program.'
    else:
        handle = Handle(**options)

    # Load the dataset
    print("Loading data...")
    dataset_options = get_args(options, ['data_id', 'padded', 'nb_categories'])
    dataset = load_dataset(**dataset_options, i_am_kfir=True)

    options['nb_categories'] = dataset_options['nb_categories']
    supervised(dataset[0], dataset[1], handle, **options)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SecS - Convolutional Motif Embeddings Tool',
                                     argument_default=argparse.SUPPRESS)

    parser.add_argument("data_id",
                        help='The protein dataset to be trained on.')

    parser.add_argument("--filters", type=int,
                        help='Number of filters in the convolutional layers.')

    parser.add_argument("--filter_length", type=int,
                        help='Size of filters in the first convolutional layer.')

    parser.add_argument("--no_pad", action='store_true',
                        help='Toggle to pad protein sequences. Batch size auto-change to 1.')

    #parser.add_argument("--mode", choices=['transfer', 'unsupervised', 'supervised'], default='unsupervised')

    parser.add_argument("--conv", type=int,
                        help='number of conv layers.')

    parser.add_argument("--fc", type=int,
                        help='number of fc layers.')

    parser.add_argument("--lstm", type=int, default=1,
                        help='number of fc layers.')

    parser.add_argument("-e", "--epochs", default=50, type=int,
                        help='number of training epochs to perform (default: 50)')

    parser.add_argument("-b", "--batch_size", type=int, default=256,
                        help='Size of minibatch.')

    parser.add_argument("--rate", type=float,
                        help='The learning rate for the optimizer.')

    parser.add_argument("--model", type=str,
                        help='Continue training the given model. Other architecture options are unused.')

    parser.add_argument("--optimizer", choices=['adam', 'nadam', 'rmsprop', 'sgd', 'adadelta', 'adagrad'],
                        help='The optimizer to be used.')

    parser.add_argument('--nb_categories', '-c', type=int, default=8,
                        help='how many categories (3/8)?')

    parser.add_argument('--dilation', type=int, default=1,
                        help='dilation?')

    parser.add_argument('--validation', type=float, default=.2,
                        help='validation set size?')

    args = parser.parse_args()

    kwargs = args.__dict__

    if hasattr(args, 'no_pad'):
        kwargs['batch_size'] = 1
        kwargs.pop('no_pad')
        kwargs['padded'] = False

    if hasattr(args, 'model'):
        kwargs.pop('filters')
        kwargs.pop('filter_length')

    main(**kwargs)

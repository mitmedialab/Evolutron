#!/usr/bin/env python
# coding=utf-8
"""
    Embedder
    ------------------------------------------_
    Embedder is an unsupervised model to generate AA embeddings.

    (c) Massachusetts Institute of Technology

    For more information contact:
    kfirs [at] mit.edu
"""

import argparse
import time

import numpy as np

import keras.backend as K

# Check if package is installed, else fallback to developer mode imports
try:
    import evolutron.networks as nets
    from evolutron.tools import load_dataset, none2str, Handle
    from evolutron.engine import DeepTrainer
    from evolutron.networks.krs.SecS import DeepEmbed
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.abspath('..'))
    import evolutron.networks as nets
    from evolutron.tools import load_dataset, none2str, Handle, shape
    from evolutron.engine import DeepTrainer
    from evolutron.networks.krs.SecS import DeepEmbed


def unsupervised(train_data, embed_data, handle,
               epochs=10,
               batch_size=128,
               filter_length=10,
               validation=.3,
               optimizer='sgd',
               rate=.01,
               conv=1,
               lstm=1,
               nb_categories=8,
               model=None,
               mode=None,
               clipnorm=0):

    filters = nb_categories

    # Find input shape
    if type(train_data) == np.ndarray:
        input_shape = train_data[0].shape
    elif type(train_data) == list:
        input_shape = (None, train_data[0].shape[1])
    else:
        raise TypeError('Something went wrong with the dataset type')

    if model:
        net_arch = DeepEmbed.from_saved_model(model)
        print('Loaded model')
    else:
        print('Building model ...')
        #train_net_arch, embed_net_arch = DeepEmbed.from_options(input_shape,
        train_net_arch = DeepEmbed.from_options(input_shape,
                                                                 n_conv_layers=conv,
                                                                 use_lstm=lstm,
                                                                 n_filters=filters,
                                                                 filter_length=filter_length,
                                                                 nb_categories=nb_categories,
                                                                 )
        handle.model = 'realDeepEmbed'

    train_net = DeepTrainer(train_net_arch)
    #train_net = DeepTrainer(embed_net_arch)
    train_net.compile(optimizer=optimizer, lr=rate, clipnorm=clipnorm, mode=mode)

    train_net.display_network_info()

    print('Started training at {}'.format(time.asctime()))

    train_net.fit(train_data, train_data,
                 nb_epoch=epochs,
                 batch_size=batch_size,
                 validate=validation,
                 patience=50,
                 )

    print('Testing model ...')
    score = train_net.score(train_data, train_data)
    print('Test Loss:{0:.6f}, Test Accuracy: {1:.2f}%'.format(score[0], 100 * score[1]))

    train_net.save_train_history(handle)
    train_net.save_model_to_file(handle)

    embed_net = K.function([train_net_arch.layers[0].input], [train_net_arch.layers[conv+1].output])
    embeddings = embed_net([embed_data])[0]

    return embeddings


def get_args(kwargs, args):
    return {k: kwargs.pop(k) for k in args if k in kwargs}


def main(**options):
    if 'model' in options:
        handle = Handle.from_filename(options.get('model'))
        #assert handle.program == 'SecS', 'The model file provided is for another program.'
    else:
        handle = Handle(**options)

    # Load the dataset
    print("Loading data...")
    train_dataset_options = get_args(options, ['train_data_id', 'padded', 'nb_categories'])
    train_dataset_options['data_id'] = train_dataset_options['train_data_id']
    del(train_dataset_options['train_data_id'])
    train_dataset, _ = load_dataset(**train_dataset_options, i_am_kfir=True)

    embed_dataset_options = get_args(options, ['embed_data_id', 'padded', 'nb_categories'])
    embed_dataset_options['data_id'] = embed_dataset_options['embed_data_id']
    del (embed_dataset_options['embed_data_id'])
    embed_dataset, _ = load_dataset(**embed_dataset_options, i_am_kfir=True, max_aa=train_dataset.shape[1])

    options['nb_categories'] = train_dataset_options['nb_categories']
    return unsupervised(train_dataset, embed_dataset, handle, **options)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embedder - Embeddings Tool',
                                     argument_default=argparse.SUPPRESS)

    parser.add_argument("train_data_id",
                        help='The protein dataset to be trained on.')

    parser.add_argument("embed_data_id",
                        help='The protein dataset to compute embeddings for.')

    parser.add_argument("--filter_length", type=int,
                        help='Size of filters in the first convolutional layer.')

    parser.add_argument("--no_pad", action='store_true',
                        help='Toggle to pad protein sequences. Batch size auto-change to 1.')

    parser.add_argument("--conv", type=int,
                        help='number of conv layers.')

    parser.add_argument("--lstm", type=int, default=1,
                        help='number of fc layers.')

    parser.add_argument("-e", "--epochs", default=50, type=int,
                        help='number of training epochs to perform (default: 50)')

    parser.add_argument("-b", "--batch_size", type=int, default=256,
                        help='Size of minibatch.')

    parser.add_argument("--rate", type=float,
                        help='The learning rate for the optimizer.')

    parser.add_argument("--clipnorm", type=float,
                        help='Clipping the gradient norm for the optimizer.')

    parser.add_argument("--model", type=str,
                        help='Continue training the given model. Other architecture options are unused.')

    parser.add_argument("--optimizer", choices=['adam', 'nadam', 'rmsprop', 'sgd', 'adadelta', 'adagrad'],
                        help='The optimizer to be used.')

    parser.add_argument('--nb_categories', '-c', type=int, default=8,
                        help='how many categories (3/8)?')

    parser.add_argument('--validation', type=float, default=.2,
                        help='validation set size?')

    parser.add_argument("--mode", choices=['NaNGuardMode', 'None'],
                        help='Theano mode to be used.')

    args = parser.parse_args()

    kwargs = args.__dict__

    if hasattr(args, 'no_pad'):
        kwargs['batch_size'] = 1
        kwargs.pop('no_pad')
        kwargs['padded'] = False

    if hasattr(args, 'model'):
        try:
            kwargs.pop('filters')
        except:
            pass
        try:
            kwargs.pop('filter_length')
        except:
            pass

    main(**kwargs)

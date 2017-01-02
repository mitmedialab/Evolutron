#!/usr/bin/env python
# coding=utf-8
"""
    DeepSecS - Deep Secondary Structure
    ------------------------------------------_
    DeepSecS is an automated tool for prediction of protein secondary structure
     from it's amino acid sequence.

    (c) Massachusetts Institute of Technology

    For more information contact:
    kfirs [at] mit.edu
"""

import argparse
import time

import numpy as np
from keras.callbacks import ReduceLROnPlateau
try:
    from .Embedder import main as Embedder
except SystemError:
    from Embedder import main as Embedder

# Check if package is installed, else fallback to developer mode imports
try:
    import evolutron.networks as nets
    from evolutron.tools import load_dataset, none2str, Handle
    from evolutron.tools.seq_tools import hot2aa, hot2SecS_8cat
    from evolutron.engine import DeepTrainer
    from evolutron.networks.krs.SecS import DeepSecS
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.abspath('..'))
    import evolutron.networks as nets
    from evolutron.tools import load_dataset, none2str, Handle, shape
    from evolutron.tools.seq_tools import hot2aa, hot2SecS_8cat
    from evolutron.engine import DeepTrainer
    from evolutron.networks.krs.SecS import DeepSecS


def supervised(x_data, y_data, handle,
               epochs=50,
               batch_size=64,
               filters=100,
               filter_length=10,
               units=500,
               validation=.3,
               optimizer='nadam',
               rate=.01,
               conv=1,
               fc=1,
               lstm=1,
               nb_categories=8,
               dilation=1,
               p=.4,
               nb_lc_units=96,
               lc_filter_length=11,
               l=.1503,
               model=None,
               mode=None,
               clipnorm=0,
               embeddings=None,
               **dataset_options):

    # Find input shape
    if type(x_data) == np.ndarray:
        input_shape = x_data[0].shape
    elif type(x_data) == list:
        input_shape = (None, x_data[0].shape[1])
    else:
        raise TypeError('Something went wrong with the dataset type')

    if model:
        net_arch = DeepSecS.from_saved_model(model)
        print('Loaded model')
    else:
        print('Building model ...')
        net_arch = DeepSecS.from_options(input_shape,
                                         n_conv_layers=conv,
                                         n_fc_layers=fc,
                                         use_lstm=lstm,
                                         n_filters=filters,
                                         filter_length=filter_length,
                                         nb_categories=nb_categories,
                                         dilation=dilation,
                                         nb_units=units,
                                         p=p,
                                         nb_lc_units=nb_lc_units,
                                         lc_filter_length=lc_filter_length,
                                         l=l)
        handle.model = 'DeepSecS'

    print('Compiling model ...')
    conv_net = DeepTrainer(net_arch)
    conv_net.display_network_info()
    conv_net.compile(optimizer=optimizer, lr=rate, clipnorm=clipnorm, mode=mode)

    print('Started training at {}'.format(time.asctime()))

    rl = ReduceLROnPlateau(factor=.2, patience=10)

    conv_net.fit(x_data, y_data,
                 nb_epoch=epochs,
                 batch_size=batch_size,
                 validate=validation,
                 patience=50,
                 extra_callbacks=[],
                 reduce_factor=.2)

    print('Testing model ...')
    score = conv_net.score(x_data, y_data)
    print('Test Loss:{0:.6f}, Test Accuracy: {1:.2f}%'.format(score[0], 100 * score[1]))

    prediction = conv_net.predict(x_data)
    with open('tmp/cullPDB_%.2f.txt' % score[0], 'w') as f:
        for i in range(x_data.shape[0]):
            f.write(hot2aa(x_data[i,:,:22]))
            f.write('\n')
            f.write(hot2SecS_8cat(y_data[i,:,:]))
            f.write('\n')
            f.write(hot2SecS_8cat(prediction[i, :, :]))
            f.write('\n')
            f.write('\n')

    print('Testing model with CB513...')
    dataset_options['pad_y_data'] = True
    x_data, y_data = load_dataset(data_id='cb513', **dataset_options)
    x_data = augment(x_data, embeddings, data_id='cb513', **dataset_options)
    cb513_score = conv_net.score(x_data, y_data)
    print('Test Loss:{0:.6f}, Test Accuracy: {1:.2f}%'.format(cb513_score[0], 100 * cb513_score[1]))

    if dataset_options['extra_features'] is None:
        print('Testing model with CASP10...')
        dataset_options.pop('extra_features')
        x_data, y_data = load_dataset(data_id='casp10', **dataset_options, min_aa=700, max_aa=700)
        x_data = augment(x_data, embeddings, data_id='casp10', **dataset_options)
        casp10_score = conv_net.score(x_data, y_data)
        print('Test Loss:{0:.6f}, Test Accuracy: {1:.2f}%'.format(casp10_score[0], 100 * casp10_score[1]))

        prediction = conv_net.predict(x_data)
        with open('tmp/casp10_%.2f.txt' % score[0], 'w') as f:
            for i in range(x_data.shape[0]):
                f.write(hot2aa(x_data[i, :, :]))
                f.write('\n')
                f.write(hot2SecS_8cat(y_data[i, :, :]))
                f.write('\n')
                f.write(hot2SecS_8cat(prediction[i, :, :]))
                f.write('\n')
                f.write('\n')

        print('Testing model with CASP11...')
        x_data, y_data = load_dataset(data_id='casp11', **dataset_options, min_aa=700, max_aa=700)
        x_data = augment(x_data, embeddings, data_id='casp11', **dataset_options)
        casp11_score = conv_net.score(x_data, y_data)
        print('Test Loss:{0:.6f}, Test Accuracy: {1:.2f}%'.format(casp11_score[0], 100 * casp11_score[1]))
    else:
        casp10_score = (0, 0)
        casp11_score = (0, 0)

    conv_net.save_train_history(handle)
    conv_net.save_model_to_file(handle)

    return score, cb513_score, casp10_score, casp11_score, handle


def get_args(kwargs, args):
    return {k: kwargs.pop(k) for k in args if k in kwargs}


def augment(x_data, embeddings, **dataset_options):
    if embeddings:
        if embeddings.find('CoMET'):
            dataset_options['nb_aa'] = 20

        if embeddings[0] == '[':
            embeddings = embeddings[1:-1].split(',')
        else:
            embeddings = [embeddings]

        for embed_model in embeddings:
            embed = Embedder(**dataset_options, mode='embed',
                             model=embed_model)
            x_data = np.concatenate((x_data, embed), axis=-1)

    return x_data


def main(**options):
    if 'model' in options:
        handle = Handle.from_filename(options.get('model'))
        #assert handle.program == 'SecS', 'The model file provided is for another program.'
    else:
        t = time.gmtime()
        """options['filters'] = str(t[0]) + str(t[1]).zfill(2) + str(t[2]).zfill(2) \
                             + str(t[3]).zfill(2) + str(t[4]).zfill(2)"""
        handle = Handle(**options)

    # Load the dataset
    print("Loading data...")
    dataset_options = get_args(options, ['data_id', 'padded', 'nb_categories', 'pssm', 'codon_table',
                                         'extra_features', 'nb_aa'])
    x_data, y_data = load_dataset(**dataset_options, pad_y_data=True)

    if 'embeddings' in options:
        x_data = augment(x_data, options['embeddings'], **dataset_options)

    dataset_options.pop('data_id')
    return supervised(x_data, y_data, handle, **options, **dataset_options)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SecS - Convolutional Motif Embeddings Tool',
                                     argument_default=argparse.SUPPRESS)

    parser.add_argument("data_id",
                        help='The protein dataset to be trained on.')

    parser.add_argument("--filters",
                        help='Number of filters in the convolutional layers.')

    parser.add_argument("--filter_length",
                        help='Size of filters in the first convolutional layer.')

    parser.add_argument("--no_pad", action='store_true',
                        help='Toggle to pad protein sequences. Batch size auto-change to 1.')

    parser.add_argument("--conv", type=int,
                        help='number of conv layers.')

    parser.add_argument("--fc", type=int,
                        help='number of fc layers.')

    parser.add_argument("--lstm", type=int, default=1,
                        help='number of fc layers.')

    parser.add_argument("-e", "--epochs", default=50, type=int,
                        help='number of training epochs to perform (default: 50)')

    parser.add_argument("-b", "--batch_size", type=int, default=64,
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

    parser.add_argument('--dilation', type=int, default=1,
                        help='dilation?')

    parser.add_argument('--validation', type=float, default=.2,
                        help='validation set size?')

    parser.add_argument("--mode", choices=['NaNGuardMode', 'None'],
                        help='Theano mode to be used.')

    parser.add_argument("--pssm", action='store_true',
                        help='Use PSSM features')

    parser.add_argument("--codon_table", action='store_true',
                        help='Use PSSM features')

    parser.add_argument("--extra_features", action='store_true',
                        help='Use PSSM features')

    parser.add_argument("--embeddings", type=str,
                        help='List of embeddings to use.')

    parser.add_argument('--nb_aa', type=int, default=22,
                        help='how many aa in the alphabet?')

    parser.add_argument('--units', type=int, default=500,
                        help='number of units in the inner network')

    parser.add_argument('--p', type=float, default=.4,
                        help='Dropout probability')

    parser.add_argument('--l', type=float, default=.1503,
                        help='l2 regulizer coefficient')

    parser.add_argument('--nb_lc_units', type=int, default=96,
                        help='number of locally connected filters')

    parser.add_argument('--lc_filter_length', type=int, default=11,
                        help='filter length for the locally connected layers')

    args = parser.parse_args()

    kwargs = args.__dict__

    if hasattr(args, 'no_pad'):
        kwargs['batch_size'] = 1
        kwargs.pop('no_pad')
        kwargs['padded'] = False
    else:
        kwargs['padded'] = True

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

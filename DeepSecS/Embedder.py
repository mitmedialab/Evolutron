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
import h5py

import numpy as np
from sklearn.preprocessing import OneHotEncoder

import keras.backend as K
from keras.models import model_from_json

# Check if package is installed, else fallback to developer mode imports
try:
    import evolutron.networks as nets
    from evolutron.tools import load_dataset, none2str, Handle
    from evolutron.engine import DeepTrainer
    from evolutron.networks.krs.Embedders import DeepCoDER, DeepEmbed, DeepCoFAM as DeepFAM
    from evolutron.networks import custom_layers
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.abspath('..'))
    import evolutron.networks as nets
    from evolutron.tools import load_dataset, none2str, Handle, shape
    from evolutron.engine import DeepTrainer
    from evolutron.networks.krs.Embedders import DeepCoDER, DeepEmbed, DeepCoFAM as DeepFAM
    from evolutron.networks import custom_layers


def train(x_data, y_data, handle,
          epochs=10,
          batch_size=64,
          filters=50,
          filter_length=15,
          validation=.2,
          optimizer='sgd',
          rate=.01,
          conv=5,
          lstm=1,
          nb_categories=8,
          model=None,
          embedder='conv',
          clipnorm=0):

    # Find input shape
    if type(x_data) == np.ndarray:
        input_shape = x_data[0].shape
    elif type(x_data) == list:
        input_shape = (None, x_data[0].shape[1])
    else:
        raise TypeError('Something went wrong with the dataset type')

    if model:
        net_arch = DeepEmbed.from_saved_model(model)
        print('Loaded model')
    else:
        print('Building model ...')

        if embedder in ['rnn', 'brnn']:
            net_arch = DeepEmbed.from_options(input_shape,
                                              n_conv_layers=conv,
                                              n_filters=filters,
                                              filter_length=filter_length,
                                              nb_categories=nb_categories,
                                              embedder=embedder)
        elif embedder == 'conv':
            net_arch = DeepCoDER.from_options(input_shape,
                                              n_conv_layers=conv,
                                              n_filters=filters,
                                              filter_length=filter_length,
                                              )
        else:
            nb_families = y_data.shape[-1]
            print(x_data.shape)
            print(y_data.shape)
            net_arch = DeepFAM.from_options(input_shape,
                                            n_conv_layers=conv,
                                            n_filters=filters,
                                            filter_length=filter_length,
                                            output_dim=nb_families,
                                            )
        handle.model = embedder+'Embedder'

    train_net = DeepTrainer(net_arch)
    train_net.compile(optimizer=optimizer, lr=rate, clipnorm=clipnorm)

    train_net.display_network_info()

    print('Started training at {}'.format(time.asctime()))

    train_net.fit(x_data, y_data,
                  nb_epoch=epochs,
                  batch_size=batch_size,
                  validate=validation,
                  patience=50,
                  )

    print('Testing model ...')
    score = train_net.score(x_data, y_data)
    print('Test Loss:{0:.6f}, Test Accuracy: {1:.2f}%'.format(score[0], 100 * score[1]))

    train_net.save_train_history(handle)
    train_net.save_model_to_file(handle)

    return score


def embed(proteins, model, data_id, **kwargs):

    embed_dataset = model.split('/')[-2]
    embed_model = (model.split('/')[-1]).split('.')[0]

    try:
        with np.load('embeddings/' + data_id + '/' + embed_model + '_' + embed_dataset + '.embed.npz') as f:
            emb = f['arr_0']

        print('Embeddings already exists')

    except IOError:
        # proteins = load_dataset(data_id, padded=True, i_am_kfir=True)

        hf = h5py.File(model)
        model_config = hf.attrs['model_config'].decode('utf8')
        hf.close()
        net = DeepTrainer(model_from_json(model_config, custom_objects=custom_layers))

        net.load_all_param_values(model)

        try:
            code_layer = [layer for layer in net.get_all_layers() if layer.name.find('encoded') == 0][-1]
        except IndexError:
            code_layer = [layer for layer in net.get_all_layers() if layer.name.find('Conv') == 0][-1]

        embed_fun = K.function(inputs=[net.input], outputs=code_layer.output)

        emb = np.asarray([embed_fun([[x]]) for x in proteins]).squeeze()

        np.savez('embeddings/' + data_id + '/' + embed_model + '_' + embed_dataset + '.embed.npz', emb)

        print('Generated embeddings')

    return emb


def get_args(kwargs, args):
    return {k: kwargs.pop(k) for k in args if k in kwargs}


def main(**options):
    if 'model' in options and options['mode'] == 'train':
        handle = Handle.from_filename(options.get('model'))
        #assert handle.program == 'SecS', 'The model file provided is for another program.'
    elif options['mode'] == 'train':
        t = time.gmtime()
        handle_options = options.copy()
        handle_options['filters'] = str(t[0]) + str(t[1]).zfill(2) + str(t[2]).zfill(2) \
                             + str(t[3]).zfill(2) + str(t[4]).zfill(2)
        handle_options['fc'] = options['embedder']
        handle = Handle(**handle_options)

    # Load the dataset
    print("Loading data...")
    dataset_options = get_args(options, ['data_id', 'padded', 'nb_aa'])
    mode = options.pop('mode')
    if mode == 'train' and options['embedder'] == 'fam':
        x_data, y_data = load_dataset(**dataset_options, codes=True, min_aa=700, max_aa=700)
        one_hot = OneHotEncoder(sparse=False)
        y_data = one_hot.fit_transform(X=np.reshape(y_data, (-1, 1)))
    else:
        try:
            x_data, _ = load_dataset(**dataset_options, min_aa=700, max_aa=700)
        except:
            x_data = load_dataset(**dataset_options, min_aa=700, max_aa=700)
        y_data = x_data

    if mode == 'train':
        return train(x_data, y_data, handle, **options)
    else:
        options['data_id'] = dataset_options['data_id']
        return embed(x_data, **options)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embedder - Embeddings Tool',
                                     argument_default=argparse.SUPPRESS)

    parser.add_argument("data_id",
                        help='The protein dataset to be trained on.')

    parser.add_argument("--filter_length", type=int,
                        help='Size of filters in the first convolutional layer.')

    parser.add_argument("--no_pad", action='store_true',
                        help='Toggle to pad protein sequences. Batch size auto-change to 1.')

    parser.add_argument("--conv", type=int, default=5,
                        help='number of conv layers.')

    parser.add_argument("--lstm", type=int, default=1,
                        help='number of lstm layers.')

    parser.add_argument("-e", "--epochs", default=200, type=int,
                        help='number of training epochs to perform (default: 50)')

    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help='Size of minibatch.')

    parser.add_argument("--rate", type=float,
                        help='The learning rate for the optimizer.')

    parser.add_argument("--clipnorm", type=float,
                        help='Clipping the gradient norm for the optimizer.')

    parser.add_argument("--model", type=str,
                        help='Model used to generate the embeddings')

    parser.add_argument("--optimizer", choices=['adam', 'nadam', 'rmsprop', 'sgd', 'adadelta', 'adagrad'],
                        default='sgd', help='The optimizer to be used.')

    parser.add_argument('--nb_categories', '-c', type=int, default=8,
                        help='how many categories (3/8)?')

    parser.add_argument('--validation', type=float, default=.1,
                        help='validation set size?')

    parser.add_argument("--mode", choices=['train', 'embed'], default='train',
                        help='mode to be used.')

    parser.add_argument("--embedder", choices=['conv', 'rnn', 'brnn', 'fam'], default='conv',
                        help='Which embedder to train.')

    parser.add_argument('--nb_aa', type=int, default=22,
                        help='how many aa in the alphabet?')

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

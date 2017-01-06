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
from sklearn.model_selection import train_test_split
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
    from evolutron.networks.krs.GAN_SecS import DeepSecS_Gen_Dis
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.abspath('..'))
    import evolutron.networks as nets
    from evolutron.tools import load_dataset, none2str, Handle, shape
    from evolutron.tools.seq_tools import hot2aa, hot2SecS_8cat
    from evolutron.engine import DeepTrainer
    from evolutron.networks.krs.GAN_SecS import DeepSecS_Gen_Dis


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
        nb_train_sample = x_data.shape[0]
    elif type(x_data) == list:
        input_shape = (None, x_data[0].shape[1])
        nb_train_sample = len(x_data)
    else:
        raise TypeError('Something went wrong with the dataset type')

    if model:
        net_arch = DeepSecS_Gen_Dis.from_saved_model(model)
        print('Loaded model')
    else:
        print('Building model ...')
        generator_arc, discriminator_arc, gen_dis_arc = DeepSecS_Gen_Dis.from_options(input_shape,
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
    generator = DeepTrainer(generator_arc)
    generator.display_network_info()
    generator.compile(optimizer=optimizer, lr=rate, clipnorm=clipnorm, mode=mode)

    discriminator = DeepTrainer(discriminator_arc)
    discriminator.display_network_info()
    discriminator.compile(optimizer=optimizer, lr=rate, clipnorm=clipnorm, mode=mode)

    gen_dis = DeepTrainer(gen_dis_arc)
    gen_dis.display_network_info()
    gen_dis.compile(optimizer=optimizer, lr=rate, clipnorm=clipnorm, mode=mode)

    print('Started training at {}'.format(time.asctime()))

    rl = ReduceLROnPlateau(factor=.2, patience=10)

    index_array = np.arange(nb_train_sample)
    train_dis = True

    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=validation, random_state=5)

    x_gen, x_dis, y_gen, y_dis = train_test_split(x_train, y_train, test_size=.5, random_state=5)
    x_gen_val, x_dis_val, y_gen_val, y_dis_val = train_test_split(x_valid, y_valid, test_size=.5, random_state=5)

    for epoch in range(1, epochs+1):
        """callbacks.on_epoch_begin(epoch)
        if shuffle == 'batch':
            index_array = batch_shuffle(index_array, batch_size)
        elif shuffle:
            np.random.shuffle(index_array)"""

        epoch_start = time.time()

        batches = make_batches(min(len(x_gen), len(x_dis)), batch_size)
        epoch_logs = {}
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            print('Epoch %d, batch %d out of %d' % (epoch, batch_index, len(batches)))
            batch_ids = index_array[batch_start:batch_end]

            # Train the generator on half the samples for classification accuracy
            generator.network.train_on_batch(x_dis[batch_ids], y_dis[batch_ids])
            generated_ys = generator.predict(x_gen[batch_ids])

            # Build the discriminator batch
            dis_x_batch = np.concatenate((np.concatenate((x_dis[batch_ids], y_dis[batch_ids]), axis=-1),
                                          np.concatenate((x_gen[batch_ids], generated_ys), axis=-1)), axis=0)
            dis_y_batch = np.array([0]*len(batch_ids) + [1]*len(batch_ids))
            dis_x_batch, dis_y_batch = shuffle_in_unison(dis_x_batch, dis_y_batch)

            # Train the discriminator (only if the accuracy on the last batch was lower then 80%)
            discriminator.network.trainable = True
            if train_dis:
                _, dis_batch_acc = discriminator.network.train_on_batch(dis_x_batch, dis_y_batch)

                if dis_batch_acc > .8:
                    train_dis = False
            else:
                _, dis_batch_acc = discriminator.network.test_on_batch(dis_x_batch, dis_y_batch)

                if dis_batch_acc < .8:
                    train_dis = True

            # Train the generator on the second half of samples for classification accuracy and discriminator error
            discriminator.network.trainable = False
            gen_dis.network.train_on_batch(x_gen[batch_ids], [y_gen[batch_ids], np.array([1]*len(batch_ids))])

            #callbacks.on_batch_end(batch_index, batch_logs)

            if batch_index == len(batches) - 1:  # last batch
                gen_loss, gen_acc = generator.score(x_valid, y_valid)

                generated_ys = generator.predict(x_gen_val)
                dis_x_valid = np.concatenate((np.concatenate((x_dis_val, y_dis_val), axis=-1),
                                              np.concatenate((x_gen_val, generated_ys), axis=-1)), axis=0)
                dis_y_valid = np.array([0] * len(x_dis_val) + [1] * len(x_gen_val))
                dis_x_valid, dis_y_valid = shuffle_in_unison(dis_x_valid, dis_y_valid)

                dis_loss, dis_acc = discriminator.score(dis_x_valid, dis_y_valid)

                print('Epoch %d: time - %.2f secs, G loss - %f, G accuracy - %.3f, D loss - %f, D accuracy - %.3f'
                      % (epoch, time.time()-epoch_start, gen_loss, gen_acc*100, dis_loss, dis_acc*100))

        """callbacks.on_epoch_end(epoch, epoch_logs)
        if callback_model.stop_training:
            break
    callbacks.on_train_end()"""

    x_valid, y_valid = generator.fit(x_data, y_data,
                                    nb_epoch=epochs,
                                    batch_size=batch_size,
                                    validate=validation,
                                    patience=50,
                                    extra_callbacks=[],
                                    reduce_factor=.2)

    print('Testing model ...')
    score = generator.score(x_valid, y_valid)
    print('Test Loss:{0:.6f}, Test Accuracy: {1:.2f}%'.format(score[0], 100 * score[1]))

    prediction = generator.predict(x_data)
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
    cb513_score = generator.score(x_data, y_data)
    print('Test Loss:{0:.6f}, Test Accuracy: {1:.2f}%'.format(cb513_score[0], 100 * cb513_score[1]))

    if not(dataset_options['extra_features'] or dataset_options['pssm'] or dataset_options['codon_table']):
        print('Testing model with CASP10...')
        dataset_options.pop('pssm')
        dataset_options.pop('codon_table')
        dataset_options.pop('extra_features')
        x_data, y_data = load_dataset(data_id='casp10', **dataset_options, min_aa=700, max_aa=700)
        x_data = augment(x_data, embeddings, data_id='casp10', **dataset_options)
        casp10_score = generator.score(x_data, y_data)
        print('Test Loss:{0:.6f}, Test Accuracy: {1:.2f}%'.format(casp10_score[0], 100 * casp10_score[1]))

        prediction = generator.predict(x_data)
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
        casp11_score = generator.score(x_data, y_data)
        print('Test Loss:{0:.6f}, Test Accuracy: {1:.2f}%'.format(casp11_score[0], 100 * casp11_score[1]))
    else:
        casp10_score = (0, 0)
        casp11_score = (0, 0)

    generator.save_train_history(handle)
    generator.save_model_to_file(handle)

    with open('ResultsDB.txt', mode='a') as file:
        file.write('%d;%d;' % (epochs, batch_size) + str(filters) + ';' + str(filter_length) +
                   ';%d;%.3f;' % (units, validation) + optimizer +
                   ';%f;%d;%d;%d;%d;%d;%.2f;%d;%d;%.5f;'
                   % (rate, conv, fc, lstm, nb_categories, dilation, p, nb_lc_units, lc_filter_length, l)
                   + str(model) +';%.3f;' % clipnorm + str(embeddings) + ';' + str(dataset_options['extra_features'])
                   + ';' + str(dataset_options['pssm']) + ';' + str(dataset_options['codon_table']) + ';' +
                   '%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f\n'
                   % (score[0], 100 * score[1], cb513_score[0], 100 * cb513_score[1], casp10_score[0],
                      100 * casp10_score[1], casp11_score[0], 100 * casp11_score[1]))
    return score, cb513_score, casp10_score, casp11_score, handle


def make_batches(size, batch_size):
    '''Returns a list of batch indices (tuples of indices).
    '''
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, nb_batch)]


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def slice_X(X, start=None, stop=None):
    '''This takes an array-like, or a list of
    array-likes, and outputs:
        - X[start:stop] if X is an array-like
        - [x[start:stop] for x in X] if X in a list

    Can also work on list/array of indices: `slice_X(x, indices)`

    # Arguments:
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.
    '''
    if isinstance(X, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return X[start]
        else:
            return X[start:stop]


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

    parser.add_argument("--filters", default=64,
                        help='Number of filters in the convolutional layers.')

    parser.add_argument("--filter_length", default='[11,7,11,15]',
                        help='Size of filters in the first convolutional layer.')

    parser.add_argument("--no_pad", action='store_true',
                        help='Toggle to pad protein sequences. Batch size auto-change to 1.')

    parser.add_argument("--conv", type=int, default=3,
                        help='number of conv layers.')

    parser.add_argument("--fc", type=int, default=0,
                        help='number of fc layers.')

    parser.add_argument("--lstm", type=int, default=0,
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

    if not(hasattr(args, 'pssm')):
        kwargs['pssm'] = False
    if not(hasattr(args, 'extra_features')):
        kwargs['extra_features'] = False
    if not(hasattr(args, 'codon_table')):
        kwargs['codon_table'] = False

    main(**kwargs)

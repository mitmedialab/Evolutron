#!/usr/bin/env python

from __future__ import print_function

import os

import predict_b1h as zfb1h
import predict_type2p as type2p


def test_type2p():
    print('\n Testing padded sequences')
    options = {
        'num_epochs': 1,
        'filters': 10,
        'filter_size': 10
    }

    type2p.main('train', True, handle='test', validation=0.2, holdout=0.1, **options)
    type2p.main('k_fold', True, handle='test', num_folds=2, **options)

    os.remove('models/type2p/test.history.npz')
    os.remove('models/type2p/test.k_fold.npz')
    os.remove('models/type2p/test.model.npz')

    print('\n Testing un-padded sequences')

    type2p.main('train', False, handle='test', validation=0.2, holdout=0.1, **options)
    type2p.main('k_fold', False, handle='test', num_folds=2, **options)

    os.remove('models/type2p/test.history.npz')
    os.remove('models/type2p/test.k_fold.npz')
    os.remove('models/type2p/test.model.npz')


def test_zfb1h():
    print('\n Testing padded sequences')
    options = {
        'num_epochs': 1,
        'filters': 10,
        'filter_size': 10
    }

    zfb1h.main('train', True, handle='test', validation=0.2, holdout=0.1, **options)
    zfb1h.main('k_fold', True, handle='test', num_folds=2, **options)

    os.remove('models/b1h/test.history.npz')
    os.remove('models/b1h/test.k_fold.npz')
    os.remove('models/b1h/test.model.npz')

    print('\n Testing un-padded sequences')

    zfb1h.main('train', False, handle='test', validation=0.2, holdout=0.1, **options)
    zfb1h.main('k_fold', False, handle='test', num_folds=2, **options)

    os.remove('models/b1h/test.history.npz')
    os.remove('models/b1h/test.k_fold.npz')
    os.remove('models/b1h/test.model.npz')


# def test_comet():
#     print("Testing padded sequences")
#     raw_data_no, _ = m6a(padded=True)  # here load or generate control set
#
#     try:
#         max_aa = raw_data_no.shape[2]
#     except AttributeError:
#         max_aa = max([len(aa) for aa in raw_data_no])
#
#     # raw_data_yes, _ = type2p(padded=padded, n_aa=raw_data_no.shape[2])  # here load dataset
#     raw_data_yes, _ = b1h(padded=True, n_aa=max_aa)
#
#     try:
#         print("Control Dataset: " + str(raw_data_no.shape))
#         print("Positive Dataset: " + str(raw_data_yes.shape))
#     except AttributeError:
#         print(len(raw_data_no))
#         print(len(raw_data_yes))
#
#     if isinstance(raw_data_no, np.ndarray):
#         raw_data = (np.concatenate((raw_data_no[:len(raw_data_yes)], raw_data_yes)),
#                     np.concatenate((np.zeros((len(raw_data_yes), 1)), np.ones((len(raw_data_yes), 1))))
#                     )
#     else:
#         raw_data = (raw_data_no[:len(raw_data_yes)] + raw_data_yes,
#                     np.concatenate((np.zeros((len(raw_data_yes), 1)), np.ones((len(raw_data_yes), 1))))
#                     )
#
#     x_data, y_data = load_dataset(raw_data, shuffled=True)
#
#     if isinstance(x_data, np.ndarray):
#         pad_size = x_data.shape[2]
#         batch_size = int(x_data.shape[2] / 2)
#     else:
#         pad_size = None
#         batch_size = 1
#
#     num_epochs = 1
#     validation = 0.2
#     holdout = 0.1
#     filters = 10
#     filter_size = 10
#     num_folds = 2
#
#     net_arch = CoMET(pad_size=pad_size,
#                      filters=filters,
#                      filter_size=filter_size)
#     conv_net = DeepTrainer(net_arch,
#                            max_epochs=500,
#                            batch_size=batch_size,
#                            pad_size=pad_size,
#                            classification=True)
#     conv_net.display_network_info()
#     conv_net.fit(x_data, y_data, num_epochs, validate=validation, holdout=holdout)
#     conv_net.save_train_history('tests/test'.format(num_epochs, filters, filter_size))
#     conv_net.save_model_to_file('tests/test'.format(num_epochs, filters, filter_size))
#     conv_net.k_fold(x_data, y_data, num_epochs, num_folds=num_folds)
#     conv_net.save_kfold_history('tests/test'.format(num_epochs, batch_size, filters, filter_size))
#
#     os.remove('tests/test.history.npz')
#     os.remove('tests/test.k_fold.npz')
#     os.remove('tests/test.model.npz')
#
#     print("Testing unpadded sequences")
#     raw_data_no, _ = m6a(padded=False)  # here load or generate control set
#
#     try:
#         max_aa = raw_data_no.shape[2]
#     except AttributeError:
#         max_aa = max([len(aa) for aa in raw_data_no])
#
#     # raw_data_yes, _ = type2p(padded=padded, n_aa=raw_data_no.shape[2])  # here load dataset
#     raw_data_yes, _ = b1h(padded=False, n_aa=max_aa)
#
#     try:
#         print("Control Dataset: " + str(raw_data_no.shape))
#         print("Positive Dataset: " + str(raw_data_yes.shape))
#     except AttributeError:
#         print(len(raw_data_no))
#         print(len(raw_data_yes))
#
#     if isinstance(raw_data_no, np.ndarray):
#         raw_data = (np.concatenate((raw_data_no[:len(raw_data_yes)], raw_data_yes)),
#                     np.concatenate((np.zeros((len(raw_data_yes), 1)), np.ones((len(raw_data_yes), 1))))
#                     )
#     else:
#         raw_data = (raw_data_no[:len(raw_data_yes)] + raw_data_yes,
#                     np.concatenate((np.zeros((len(raw_data_yes), 1)), np.ones((len(raw_data_yes), 1))))
#                     )
#
#     x_data, y_data = load_dataset(raw_data, shuffled=True)
#
#     if isinstance(x_data, np.ndarray):
#         pad_size = x_data.shape[2]
#         batch_size = int(x_data.shape[2] / 2)
#     else:
#         pad_size = None
#         batch_size = 1
#
#     num_epochs = 1
#     validation = 0.2
#     holdout = 0.1
#     filters = 10
#     filter_size = 10
#     num_folds = 2
#
#     net_arch = CoMET(pad_size=pad_size,
#                      filters=filters,
#                      filter_size=filter_size)
#     conv_net = DeepTrainer(net_arch,
#                            max_epochs=500,
#                            batch_size=batch_size,
#                            pad_size=pad_size,
#                            classification=True)
#     conv_net.display_network_info()
#     conv_net.fit(x_data, y_data, num_epochs, validate=validation, holdout=holdout)
#     conv_net.save_train_history('tests/test'.format(num_epochs, filters, filter_size))
#     conv_net.save_model_to_file('tests/test'.format(num_epochs, filters, filter_size))
#     conv_net.k_fold(x_data, y_data, num_epochs, num_folds=num_folds)
#     conv_net.save_kfold_history('tests/test'.format(num_epochs, batch_size, filters, filter_size))
#
#     os.remove('tests/test.history.npz')
#     os.remove('tests/test.k_fold.npz')
#     os.remove('tests/test.model.npz')

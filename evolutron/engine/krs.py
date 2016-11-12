# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import os
import time
from collections import OrderedDict, defaultdict

from tabulate import tabulate
import numpy as np

from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
import keras.optimizers as opt

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

# TODO: implement enhanced progbar logger


class DeepTrainer:
    def __init__(self,
                 network,
                 classification=False,
                 verbose=False,
                 patience=True):

        self.verbose = verbose

        self.network = network
        self.inp = network.input

        self.classification = classification

        self.patience = patience

        self.history = None
        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []
        # self.val_prc_auc_mem = []
        # self.val_roc_auc_mem = []

        self.fold_train_losses = None
        self.fold_val_losses = None
        self.k_fold_history = defaultdict(list)

        self._funcs_init = False

    def compile(self, optimizer, **options):

        opts = {'sgd': opt.SGD(lr=options.get('lr', .01),
                               decay=options.get('decay', 1e-6),
                               momentum=options.get('momentum', 0.9), nesterov=True),
                'rmsprop': opt.RMSprop(lr=options.get('lr', .001)),
                'adadelta': opt.Adadelta(lr=options.get('lr', 1.)),
                'adagrad': opt.Adagrad(lr=options.get('lr', .01)),
                'adam': opt.Adam(lr=options.get('lr', .001)),
                'nadam': opt.Nadam(lr=options.get('lr', .002)),
                'adamax': opt.Adamax(lr=options.get('lr', .002))
                }

        self.network.compile(loss=self.network._loss_function,
                             optimizer=opts[optimizer],
                             metrics=[self.network.mean_cat_acc])
        self._create_functions()

    def _create_functions(self):
        if getattr(self, '_funcs_init', True):
            return

        self.train_fn = self.network._make_train_function()
        self.predict_fn = self.network._make_predict_function()
        self.test_fn = self.network._make_test_function()

        self._funcs_init = True

    def fit_generator(self):
        # TODO: if dataset is big
        # fit_generator(self, generator, samples_per_epoch, nb_epoch, verbose=1, callbacks=[], validation_data=None,
        # nb_val_samples=None, class_weight={}, max_q_size=10, nb_worker=1, pickle_safe=False)
        raise NotImplementedError

    def fit(self, x_data, y_data, nb_epoch=1, batch_size=64, shuffle=True, validate=.0, patience=10,
            return_best_model=True):
        assert (validate >= 0)
        assert self._funcs_init

        if self.classification:
            x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=validate, stratify=y_data)
        else:
            x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=validate)

        if self.classification:
            msg = 'Distribution of Examples per set'
            print(msg)
            print('-' * len(msg))
            classes = ['Class ' + str(i) for i in range(len(np.unique(y_data)))]
            counts = dict()
            _, c = np.unique(y_train, return_counts=True)
            counts['train'] = c
            _, c = np.unique(y_valid, return_counts=True)
            counts['valid'] = c

            print(tabulate([['Set'] + classes + ['Total'],
                            ['Train'] + counts['train'].tolist() + [counts['train'].sum()],
                            ['Valid'] + counts['valid'].tolist() + [counts['valid'].sum()]],
                           stralign='center',
                           headers="firstrow"))

        # Callbacks
        es = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=patience / 2, min_lr=0.001)
        rn = np.random.random()
        checkpoint = ModelCheckpoint('/tmp/best_{0}.h5'.format(rn), monitor='val_loss', verbose=1, mode='min',
                                     save_best_only=True, save_weights_only=True)

        start_time = time.time()
        try:
            self.network.fit(x_train, y_train,
                             validation_data=(x_valid, y_valid),
                             shuffle=shuffle,
                             nb_epoch=nb_epoch,
                             batch_size=batch_size,
                             callbacks=[es, reduce_lr, checkpoint],
                             verbose=1)
        except KeyboardInterrupt:
            pass

        if return_best_model:
            self.load_all_param_values('/tmp/best_{0}.h5'.format(rn))

        self.history = self.network.history

        self.train_loss = self.history.history['loss']
        if validate > 0:
            self.valid_loss = self.history.history['val_loss']

        if self.classification:
            self.train_acc = self.history.history['acc']
            if validate > 0:
                self.valid_acc = self.history.history['val_acc']

        print(
            'Model trained for {0} epochs. Total time: {1:.3f}s'.format(len(self.train_loss), time.time() - start_time))

    def k_fold(self, x_data, y_data, epochs=1, num_folds=10, stratify=False):

        self._create_functions()

        if self.classification and stratify:
            folds = StratifiedKFold(y_data, n_folds=num_folds)
        else:
            folds = KFold(len(x_data), n_folds=num_folds)

        self.fold_val_losses = np.zeros((num_folds, epochs))
        self.fold_train_losses = np.zeros((num_folds, epochs))

        foldit = 0
        for train_idx, valid_idx in folds:
            print('Starting Fold {0}'.format(foldit + 1))

            if isinstance(x_data, np.ndarray):
                train_set_x = x_data.take(train_idx, axis=0)
                valid_set_x = x_data.take(valid_idx, axis=0)
            else:
                train_set_x = [x_data[i] for i in train_idx]
                valid_set_x = [x_data[i] for i in valid_idx]

            train_set_y = y_data.take(train_idx, axis=0)
            valid_set_y = y_data.take(valid_idx, axis=0)

            if self.classification:
                msg = 'Distribution of Examples per set'
                print(msg)
                print('-' * len(msg))
                classes = ['Class ' + str(i) for i in range(len(np.unique(y_data)))]
                counts = dict()
                _, c = np.unique(train_set_y, return_counts=True)
                counts['train'] = c
                _, c = np.unique(valid_set_y, return_counts=True)
                counts['valid'] = c

                print(tabulate([['Set'] + classes + ['Total'],
                                ['Train'] + counts['train'].tolist() + [counts['train'].sum()],
                                ['Valid'] + counts['valid'].tolist() + [counts['valid'].sum()]],
                               stralign='center',
                               headers="firstrow"))
            else:
                print('Train:{0}, Valid:{1}'.format(len(train_idx), len(valid_idx)))

            # Early-stopping parameters
            patience = 50 * len(train_set_x)  # look as this many examples regardless
            patience_increase = 2  # wait this times much longer when a new best is found
            improvement_threshold = 0.998  # a relative improvement of this much is considered significant
            best_validation_loss = np.inf
            it = 0
            done_looping = False

            epochs = min(epochs, self.max_epochs)
            epoch = 0
            start_time = time.time()
            print(tabulate([['Epoch', 'Train Loss', 'Val Loss', 'Time']], stralign='center', headers="firstrow"))
            # We iterate over epochs:
            while epoch < epochs and not done_looping:
                epoch += 1

                # In each epoch, we do a full pass over the training data:
                train_err = []
                train_acc = []
                epoch_time = time.time()
                for X, y in self._iterate_minibatches(train_set_x, train_set_y, self.batch_size):
                    # print(self.f(X, y))
                    err, acc = self.train_fn(X, y)
                    train_err.append(err)
                    train_acc.append(acc)

                # After that, we do a full pass over the validation data:
                val_err = []
                val_acc = []
                for X, y in self._iterate_minibatches(valid_set_x, valid_set_y, self.batch_size):
                    err, acc = self.val_fn(X, y)
                    val_err.append(err)
                    val_acc.append(acc)

                self.fold_train_losses[foldit, epoch - 1] = (np.mean(train_err))
                self.fold_val_losses[foldit, epoch - 1] = (np.mean(val_err))
                print(tabulate([['Epoch', 'Train Loss', 'Val Loss', 'Time'],
                                ["{}/{}".format(epoch, epochs),
                                 self.fold_train_losses[foldit, epoch - 1],
                                 self.fold_val_losses[foldit, epoch - 1],
                                 "{:.3f}s".format(time.time() - epoch_time)]],
                               tablefmt='plain', floatfmt=".6f", stralign='center',
                               headers="firstrow").rsplit('\n', 1)[-1])

                # If we have the best validation score until now
                if self.fold_val_losses[foldit, epoch - 1] < best_validation_loss:

                    # Increase patience if loss improvement is good enough
                    if self.fold_val_losses[foldit, epoch - 1] < best_validation_loss * improvement_threshold:
                        patience = max(patience, it * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = self.fold_val_losses[foldit, epoch - 1]

                # Stop if above early stopping limit
                it = (epoch - 1) * len(train_set_x)
                if patience <= it:
                    done_looping = True

            foldit += 1
            print(
                'Fold {2}: Model trained for {0} epochs. Total time: {1:.3f}s'.format(epochs, time.time() - start_time,
                                                                                      foldit))
            val_preds = self.predict_proba(valid_set_x)
            train_preds = self.predict_proba(train_set_x)
            self.k_fold_history['train_classes'].append(train_set_y)
            self.k_fold_history['val_classes'].append(valid_set_y)
            self.k_fold_history['train_preds'].append(train_preds)
            self.k_fold_history['val_preds'].append(val_preds)
            self.reset_all_param_values()

    def score(self, x_data, y_data, **options):
        return self.network.evaluate(x_data, y_data, verbose=options.pop('verbose', 0), **options)

    def predict_proba(self, x_data):
        return self.network.predict_proba(x_data)

    def predict_classes(self, x_data):
        return self.network.predict_classes(x_data)

    def predict(self, x_data):
        return self.network.predict(x_data)

    def display_network_info(self):

        print("Neural Network has {0} trainable parameters".format(self.n_params))

        layers = self.get_all_layers()

        ids = list(range(len(layers)))

        names = [layer.name for layer in layers]

        shapes = ['x'.join(map(str, layer.output_shape[1:])) for layer in layers]

        params = [layer.count_params() for layer in layers]

        tabula = OrderedDict([('#', ids), ('Name', names), ('Shape', shapes), ('Parameters', params)])

        print(tabulate(tabula, 'keys'))

    def get_all_layers(self):
        return self.network.layers

    def get_all_params(self, trainable=True):
        if trainable:
            return self.network.weights
        else:
            return self.network.non_trainable_weights

    @property
    def n_params(self):
        return self.network.count_params()

    def get_all_param_values(self):
        return self.network.get_weights()

    def set_all_param_values(self, weights):
        try:
            self.network.set_weights(weights)
        except Exception:
            msg = 'Incorrect parameter list'
            raise ValueError(msg)

    def load_all_param_values(self, filepath):
        try:
            self.network.load_weights(filepath)
        except Exception:
            msg = 'Incorrect parameter list'
            raise ValueError(msg)

    def get_conv_layers(self):
        try:
            return [x for x in self.get_all_layers() if x.name.find('Conv') == 0]
        except:
            raise AttributeError("Model has no convolutional layers.")

    def get_last_conv_layer(self):
        try:
            return [x for x in self.get_all_layers() if x.name.find('Conv') == 0][-1]
        except:
            raise AttributeError("Model has no convolutional layers.")

    def get_conv_param_values(self):
        try:
            return [x.get_weights() for x in self.get_all_layers() if x.name.find('Conv') == 0]
        except:
            raise AttributeError("Model has no convolutional layers.")

    def set_conv_param_values(self, source):

        # conv_layer = [x for x in self.get_all_layers() if x.name.find('Conv') > -1]
        #
        # if not conv_layer:
        #     raise AttributeError("Model has no convolutional layers.")
        #
        # try:
        #
        #     lasagne.layers.set_all_param_values(conv_layer[-1], source)
        # except Exception:
        #     msg = 'Incorrect parameter list'
        #     raise ValueError(msg)
        raise NotImplementedError

    def reset_all_param_values(self):
        # old = lasagne.layers.get_all_param_values(self.network)
        # new = []
        # for layer in old:
        #     shape = layer.shape
        #     if len(shape) < 2:
        #         shape = (shape[0], 1)
        #     W = lasagne.init.GlorotUniform()(shape)
        #     if W.shape != layer.shape:
        #         W = np.squeeze(W, axis=1)
        #     new.append(W)
        # self.set_all_param_values(new)
        raise NotImplementedError

    def save_model_to_file(self, handle):

        handle.ftype = 'model'
        handle.epochs = len(self.history.epoch)

        filename = 'models/' + handle
        if not os.path.exists('/'.join(filename.split('/')[:-1])):
            os.makedirs('/'.join(filename.split('/')[:-1]))

        self.network.save(filename)

        print('Model saved to: ' + filename)

    def load_model_from_file(self, filename):
        self.network = self.network.__class__.from_saved_model(filename)
        self.history = self.network.history

    def save_train_history(self, handle):
        assert (not self.train_loss == [])
        handle.ftype = 'history'
        handle.epochs = len(self.train_loss)

        filename = 'models/' + handle
        if not os.path.exists('/'.join(filename.split('/')[:-1])):
            os.makedirs('/'.join(filename.split('/')[:-1]))

        np.savez_compressed(filename,
                            train_loss=self.train_loss, val_loss=self.valid_loss,
                            train_acc=self.train_acc, val_acc_mem=self.valid_acc)
        # val_prc_auc_mem=self.val_prc_auc_mem, val_roc_auc_mem=self.val_roc_auc_mem)

        print('History saved to: ' + filename)

        # def save_kfold_history(self, filename):
        #     assert (not self.fold_train_losses[0, 0] == .0)
        #     filename_ = filename + '.k_fold'
        #     np.savez_compressed(filename_,
        #                         fold_train_losses=self.fold_train_losses,
        #                         fold_val_losses=self.fold_val_losses,
        #                         val_pred=self.k_fold_history['val_preds'],
        #                         train_pred=self.k_fold_history['train_preds'],
        #                         val_class=self.k_fold_history['val_classes'],
        #                         train_class=self.k_fold_history['train_classes'])

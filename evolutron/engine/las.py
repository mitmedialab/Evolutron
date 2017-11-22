# -*- coding: utf-8 -*-
from __future__ import print_function

import time
import os
from collections import OrderedDict, defaultdict
from functools import reduce

import lasagne
import numpy as np
import theano
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from tabulate import tabulate


class DeepTrainer:
    def __init__(self,
                 net,
                 update_fn=lasagne.updates.nesterov_momentum,
                 classification=False,
                 max_epochs=2000,
                 verbose=False,
                 batch_size=1,
                 learning_rate=0.01,
                 patience=True):

        self.verbose = verbose

        self.net = net
        self.inp = net.inp
        self.targets = net.targets
        self.network = net.network

        self.classification = classification
        self.update_fn = update_fn
        self.update_args = {'learning_rate': learning_rate,
                            'momentum': 0.975}

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience

        self.train_err_mem = []
        self.val_err_mem = []
        self.val_acc_mem = []
        self.val_prc_auc_mem = []
        self.val_roc_auc_mem = []
        self.train_acc_mem = []

        self.fold_val_losses = None
        self.fold_train_losses = None
        self.k_fold_history = defaultdict(list)

        self._funcs_init = False
        self._create_functions()

    def _create_functions(self):
        if getattr(self, '_funcs_init', True):
            return

        inputs = self.inp.values()

        targets = self.targets.values()

        loss, acc, prediction = self.net.build_loss(deterministic=False)

        val_loss, val_acc, val_prediction = self.net.build_loss(deterministic=True)

        params = self.get_all_params(trainable=True)

        updates = self.update_fn(loss, params, **self.update_args)

        # Compile a function performing a training step on a mini-batch
        self.train_fn = theano.function(inputs=inputs + targets,
                                        outputs=[loss, acc],
                                        updates=updates,
                                        allow_input_downcast=False)

        # Compile a second function computing the validation loss and accuracy: (difference: Deterministic=True)
        self.val_fn = theano.function(inputs=inputs + targets,
                                      outputs=[val_loss, val_acc],
                                      allow_input_downcast=False)

        self.pred_fn = theano.function(inputs=inputs,
                                       outputs=[val_prediction],
                                       allow_input_downcast=False)

        self.val_pred = theano.function(inputs=inputs + targets,
                                        outputs=[val_loss, val_acc, val_prediction],
                                        allow_input_downcast=False)

        # self.f = theano.function([inputs],
        #                          [lasagne.layers.get_output(self.net.layers['conv'])])
        # mode=NanGuardMode(),_
        self._funcs_init = True

    def motif_fun(self):

        inputs = self.inp.values()

        conv_scores = lasagne.layers.get_output(self.get_last_conv_layer())

        return theano.function(inputs, conv_scores)

    @staticmethod
    def _split_dataset(data_size, holdout, validate):  # TODO: Check scikit learn implementation

        test_idx = range(0, int(holdout * data_size))

        valid_idx = range(int(holdout * data_size), int((validate + holdout) * data_size))

        train_idx = range(int((validate + holdout) * data_size), data_size)

        return train_idx, valid_idx, test_idx

    @staticmethod
    def _iterate_minibatches(X, y, batch_size):
        assert len(X) == len(y)

        if batch_size == 1:
            for idx in range(0, len(X)):
                yield [X[idx]], [y[idx]]
        else:
            for start_idx in range(0, len(X), batch_size):
                excerpt = slice(start_idx, min(start_idx + batch_size, len(X)))
                yield X[excerpt], y[excerpt]  # TODO: yield excerpt

    @staticmethod
    def _iterate_huge(X, y, block_size):
        # TODO: here will implement shared variable storage and pass indexes
        assert len(X) == len(y)

        for start_idx in range(0, len(X), block_size):
            excerpt = slice(start_idx, min(start_idx + block_size, len(X)))
            yield X[excerpt], y[excerpt]

    def fit(self, x_data, y_data, epochs=1, holdout=.0, validate=.0):
        assert (validate >= 0 and holdout >= 0)
        assert self._funcs_init

        train_idx, valid_idx, test_idx = self._split_dataset(len(x_data), holdout, validate)

        if isinstance(x_data, np.ndarray):
            train_set_x = x_data.take(train_idx, axis=0)
            valid_set_x = x_data.take(valid_idx, axis=0)
            test_set_x = x_data.take(test_idx, axis=0)
        else:
            train_set_x = [x_data[i] for i in train_idx]
            valid_set_x = [x_data[i] for i in valid_idx]
            test_set_x = [x_data[i] for i in test_idx]

        if isinstance(y_data, np.ndarray):
            train_set_y = y_data.take(train_idx, axis=0)
            valid_set_y = y_data.take(valid_idx, axis=0)
            test_set_y = y_data.take(test_idx, axis=0)
        else:
            train_set_y = [y_data[i] for i in train_idx]
            valid_set_y = [y_data[i] for i in valid_idx]
            test_set_y = [y_data[i] for i in test_idx]

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
            _, c = np.unique(test_set_y, return_counts=True)
            counts['test'] = c

            print(tabulate([['Set'] + classes + ['Total'],
                            ['Train'] + counts['train'].tolist() + [counts['train'].sum()],
                            ['Valid'] + counts['valid'].tolist() + [counts['valid'].sum()],
                            ['Test'] + counts['test'].tolist() + [counts['test'].sum()]],
                           stralign='center',
                           headers="firstrow"))
        else:
            print('Train:{0}, Valid:{1}, Test:{2}'.format(len(train_idx), len(valid_idx), len(test_idx)))

        # Early-stopping parameters
        patience = 200 * len(train_set_x)  # look as this many examples regardless
        patience_increase = 2  # wait this times much longer when a new best is found
        improvement_threshold = 0.998  # a relative improvement of this much is considered significant
        best_validation_loss = np.inf
        it = 0
        done_looping = False

        epochs = min(epochs, self.max_epochs)
        epoch = 0
        start_time = time.time()
        print(tabulate([['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Time']], stralign='center',
                       headers="firstrow"))
        # We iterate over epochs:
        try:
            while epoch < epochs and not done_looping:
                epoch += 1

                # In each epoch, we do a full pass over the training data:
                train_err = []
                train_acc = []
                epoch_time = time.time()
                # count=0

                for X, y in self._iterate_minibatches(train_set_x, train_set_y, self.batch_size):

                    try:
                        err, acc = self.train_fn(X, y)
                        train_err.append(err)
                        train_acc.append(acc)
                    except ValueError:
                        print(X[0].shape)
                        pass
                    # print (self.f(X))
                    # print(pred)
                    # print(targ)
                    # print(pred.shape)
                    # print(targ.shape)
                    # count += 1
                    # print(count)

                # After that, we do a full pass over the validation data:
                val_err = []
                val_acc = []
                for X, y in self._iterate_minibatches(valid_set_x, valid_set_y, self.batch_size):
                    try:
                        err, acc = self.val_fn(X, y)
                        val_err.append(err)
                        val_acc.append(acc)
                    except MemoryError:
                        print(X[0].shape)
                        pass

                self.train_err_mem.append(np.mean(train_err))
                self.train_acc_mem.append(100 * np.mean(train_acc))
                if validate > 0:
                    self.val_err_mem.append(np.mean(val_err))
                    self.val_acc_mem.append(100 * np.mean(val_acc))
                else:
                    self.val_err_mem.append('-')
                    self.val_acc_mem.append('-')

                print(tabulate([['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Time'],
                                ["{}/{}".format(epoch, epochs),
                                 self.train_err_mem[-1], self.train_acc_mem[-1],
                                 self.val_err_mem[-1], self.val_acc_mem[-1],
                                 "{:.3f}s".format(time.time() - epoch_time)]],
                               tablefmt='plain', floatfmt=".6f", stralign='center',
                               headers="firstrow").rsplit('\n', 1)[-1])

                # If we have the best validation score until now
                if self.val_err_mem[-1] < best_validation_loss:

                    # Increase patience if loss improvement is good enough
                    if self.val_err_mem[-1] < best_validation_loss * improvement_threshold:
                        patience = max(patience, it * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = self.val_err_mem[-1]

                # print(self.get_all_param_values())

                # Stop if above early stopping limit
                it = (epoch - 1) * len(train_set_x)
                if patience <= it and self.patience:
                    done_looping = True
        except KeyboardInterrupt:
            print('Model trained for {0} epochs. Total time: {1:.3f}s'.format(epoch, time.time() - start_time))
            ct = time.ctime()
            self.save_train_history('stopped_{0}'.format(ct))
            self.save_model_to_file('stopped_{0}'.format(ct))
            exit(0)

        print('Model trained for {0} epochs. Total time: {1:.3f}s'.format(epoch, time.time() - start_time))

        if holdout > 0:
            test_loss, test_acc = self.score(test_set_x, test_set_y)
            if self.classification:
                print('Test loss: {0:.6f}\tTest accuracy: {1:.6f}'.format(test_loss, test_acc))
            else:
                print('Test loss: {0:.6f}'.format(test_loss))

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

    def score(self, X, y):
        self._create_functions()
        test_err = []
        test_acc = []
        for X, y in self._iterate_minibatches(X, y, self.batch_size):
            err, acc = self.val_fn(X, y)
            test_err.append(err)
            test_acc.append(acc)

        return np.mean(test_err), 100 * np.mean(test_acc)

    def predict_proba(self, X):
        # TODO: wtf is this ?
        self._create_functions()
        preds = []
        for X, y in self._iterate_minibatches(X, X, self.batch_size):
            preds.append(np.vstack(self.pred_fn(X)))
        return np.vstack(preds).squeeze()

    def predict(self, X):
        if not self.classification:
            return self.predict_proba(X)
        else:
            y_pred = np.argmax(self.predict_proba(X), axis=1)
            return y_pred

    def display_network_info(self):

        print("Neural Network has {0} trainable parameters".format(self.n_params))

        layers = lasagne.layers.get_all_layers(self.network)

        ids = list(range(len(layers)))

        names = [layer.name for layer in layers]

        shapes = ['x'.join(map(str, layer.output_shape[1:])) for layer in layers]

        tabula = OrderedDict([('#', ids), ('Name', names), ('Shape', shapes)])

        print(tabulate(tabula, 'keys'))

    def get_all_layers(self):
        return lasagne.layers.get_all_layers(self.network)

    def get_all_params(self, trainable=True):
        return lasagne.layers.get_all_params(self.network, trainable=trainable)

    @property
    def n_params(self):
        return np.sum(reduce(np.multiply, param.get_value().shape) for param in self.get_all_params() if param)

    def get_all_param_values(self):
        return lasagne.layers.get_all_param_values(self.network)

    def set_all_param_values(self, source):

        if isinstance(source, list):
            try:
                lasagne.layers.set_all_param_values(self.network, source)
            except Exception:
                msg = 'Incorrect parameter list'
                raise ValueError(msg)
        elif isinstance(source, str):
            try:
                f = np.load(source)
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                lasagne.layers.set_all_param_values(self.network, param_values)
            except Exception:
                msg = 'Incompatible model file'
                raise IOError(msg)
        else:
            raise AssertionError('incorrect argument \"source\"')

    def get_last_conv_layer(self):
        try:
            return [x for x in self.get_all_layers() if x.name.find('Conv') > -1][-1]
        except:
            raise AttributeError("Model has no convolutional layers.")

    def get_conv_param_values(self):
        return lasagne.layers.get_all_param_values(self.get_last_conv_layer())

    def set_conv_param_values(self, source):

        conv_layer = [x for x in self.get_all_layers() if x.name.find('Conv') > -1]

        if not conv_layer:
            raise AttributeError("Model has no convolutional layers.")

        if isinstance(source, list):
            try:

                lasagne.layers.set_all_param_values(conv_layer[-1], source)
            except Exception:
                msg = 'Incorrect parameter list'
                raise ValueError(msg)
        elif isinstance(source, str):
            try:
                f = np.load(source)
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                lasagne.layers.set_all_param_values(conv_layer[-1], param_values[:len(conv_layer) * 2 - 1])
            except Exception:
                msg = 'Incompatible model file'
                raise IOError(msg)
        else:
            raise AssertionError('incorrect argument \"source\"')

    def reset_all_param_values(self):
        old = lasagne.layers.get_all_param_values(self.network)
        new = []
        for layer in old:
            shape = layer.shape
            if len(shape) < 2:
                shape = (shape[0], 1)
            W = lasagne.init.GlorotUniform()(shape)
            if W.shape != layer.shape:
                W = np.squeeze(W, axis=1)
            new.append(W)
        self.set_all_param_values(new)

    def save_model_to_file(self, handle):

        handle.ftype = 'model'
        handle.epochs = len(self.train_err_mem)

        filename = 'models/' + handle
        if not os.path.exists('/'.join(filename.split('/')[:-1])):
            os.makedirs('/'.join(filename.split('/')[:-1]))

        params = self.get_all_param_values()

        np.savez_compressed(filename, *params)

        return 'Model saved in:' + filename

    def load_model_from_file(self, filename):
        self.set_all_param_values(filename)

    def save_train_history(self, handle):
        assert (not self.train_err_mem == [])
        handle.ftype = 'history'
        handle.epochs = len(self.train_err_mem)

        filename = 'models/' + handle
        if not os.path.exists('/'.join(filename.split('/')[:-1])):
            os.makedirs('/'.join(filename.split('/')[:-1]))

        np.savez_compressed(filename,
                            train_err_mem=self.train_err_mem, val_err_mem=self.val_err_mem,
                            train_acc_mem=self.train_acc_mem, val_acc_mem=self.val_acc_mem,
                            val_prc_auc_mem=self.val_prc_auc_mem, val_roc_auc_mem=self.val_roc_auc_mem)

    def load_train_history(self, filename):
        assert (self.train_err_mem == [])
        filename += '.history.npz'
        with np.load(filename) as f:
            self.train_err_mem = f['train_err_mem'].tolist()
            self.train_acc_mem = f['train_acc_mem'].tolist()
            self.val_err_mem = f['val_err_mem'].tolist()
            self.val_acc_mem = f['val_acc_mem'].tolist()
            self.val_prc_auc_mem = f['val_prc_auc_mem'].tolist()
            self.val_roc_auc_mem = f['val_roc_auc_mem'].tolist()

    def save_kfold_history(self, filename):
        assert (not self.fold_train_losses[0, 0] == .0)
        filename_ = filename + '.k_fold'
        np.savez_compressed(filename_,
                            fold_train_losses=self.fold_train_losses,
                            fold_val_losses=self.fold_val_losses,
                            val_pred=self.k_fold_history['val_preds'],
                            train_pred=self.k_fold_history['train_preds'],
                            val_class=self.k_fold_history['val_classes'],
                            train_class=self.k_fold_history['train_classes'])

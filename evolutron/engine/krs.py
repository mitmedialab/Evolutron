# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import time
from collections import OrderedDict, defaultdict

import numpy as np
from tabulate import tabulate

import keras.backend as K
import keras.optimizers as opt
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split

if K.backend() == 'theano':
    from theano.compile.nanguardmode import NanGuardMode
    from theano.compile.monitormode import MonitorMode


class DeepTrainer:
    def __init__(self,
                 network,
                 classification=False,
                 verbose=False,
                 generator=None,
                 nb_inputs=1,
                 nb_outputs=1):

        self.verbose = verbose
        self.generator = generator
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs

        self.network = network
        self.network.generator = generator
        try:
            self.input = network.input
            self.output = network.output
        except AttributeError:
            self.input = network.inputs
            self.output = network.outputs

        self.classification = classification

        self.history = None
        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []

        self.fold_train_losses = None
        self.fold_val_losses = None
        self.k_fold_history = defaultdict(list)

        self._funcs_init = False

    def compile(self, optimizer, **options):

        opts = {'sgd': opt.SGD(lr=options.get('lr', .01),
                               decay=options.get('decay', 1e-6),
                               momentum=options.get('momentum', 0.9), nesterov=True,
                               clipnorm=options.get('clipnorm', 0)),
                'rmsprop': opt.RMSprop(lr=options.get('lr', .001)),
                'adadelta': opt.Adadelta(lr=options.get('lr', 1.)),
                'adagrad': opt.Adagrad(lr=options.get('lr', .01)),
                'adam': opt.Adam(lr=options.get('lr', .001)),
                'nadam': opt.Nadam(lr=options.get('lr', .002),
                                   clipnorm=options.get('clipnorm', 0)),
                'adamax': opt.Adamax(lr=options.get('lr', .002))
                }

        mode = options.get('mode', None)
        if K.backend() == 'theano' and mode:
            modes = {'NaNGuardMode': NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
                     'MonitorMode': MonitorMode(pre_func=inspect_inputs, post_func=inspect_outputs),
                     }
            mode = modes[mode]

            self.network.compile(loss=options.get('loss_function', self.network._loss_function),
                                 loss_weights=options.get('loss_weights', None),
                                 optimizer=opts[optimizer],
                                 metrics=self.network.metrics,
                                 mode=mode)
        else:
            self.network.compile(loss=options.get('loss_function', self.network._loss_function),
                                 loss_weights=options.get('loss_weights', None),
                                 optimizer=opts[optimizer],
                                 metrics=self.network.metrics)

        self._create_functions()

    def _create_functions(self):
        if getattr(self, '_funcs_init', True):
            return

        self.train_fn = self.network._make_train_function()
        self.predict_fn = self.network._make_predict_function()
        self.test_fn = self.network._make_test_function()

        self._funcs_init = True

    def fit_generator(self, x_data, y_data, generator=None, nb_epoch=1, batch_size=64, shuffle=True,
                      validate=.0, patience=10, return_best_model=True, verbose=1, extra_callbacks=None,
                      reduce_factor=.5, **generator_options):

        # Check arguments
        if generator is None:
            generator = self.generator
        if extra_callbacks is None:
            extra_callbacks = []
        assert (validate >= 0)
        assert nb_epoch > 0

        if self.classification:
            stratify = y_data  # TODO: here you should select with which part to stratify
        else:
            stratify = None

        if self.nb_inputs == 1:
            x_train, x_valid = self._check_and_split_data(x_data, self.input, validate, stratify)
        else:
            stratify = None
            x_train = [[] for _ in x_data]
            x_valid = [[] for _ in x_data]
            for i, x_d in enumerate(x_data):
                x_train[i], x_valid[i] = self._check_and_split_data(x_d, self.network.inputs[i],
                                                                    validate, stratify)
        if self.nb_outputs == 1:
            y_train, y_valid = self._check_and_split_data(y_data, self.output, validate, stratify)
        else:
            stratify = None
            y_train = [[] for _ in y_data]
            y_valid = [[] for _ in y_data]
            for i, y_d in enumerate(y_data):
                y_train[i], y_valid[i] = self._check_and_split_data(y_d, self.output[i], validate, stratify)
        # Callbacks
        es = EarlyStopping(monitor='val_loss',
                           min_delta=0.0001,
                           patience=patience,
                           verbose=1,
                           mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=reduce_factor,
                                      patience=5,
                                      min_lr=0.001,
                                      verbose=1)
        rn = np.random.random()
        checkpoint = ModelCheckpoint('/tmp/best_{0}.h5'.format(rn),
                                     monitor='val_loss',
                                     verbose=1,
                                     mode='min',
                                     save_best_only=True,
                                     save_weights_only=True)

        if K.backend() == "tensorflow":
            tb = TensorBoard()
            callbacks = [es, reduce_lr, checkpoint, tb]
        else:
            callbacks = [es, reduce_lr, checkpoint]

        if self.nb_inputs == 1:
            nb_train_samples = len(x_train)
            nb_val_samples = len(x_valid)
        else:
            nb_train_samples = len(x_train[0])
            nb_val_samples = len(x_valid[0])

        for cb in extra_callbacks:
            cb.validation_data = (x_valid, y_valid)

        start_time = time.time()
        try:
            self.network.fit_generator(generator=generator(x_train, y_train, batch_size=batch_size,
                                                           shuffle=shuffle, **generator_options),
                                       steps_per_epoch=np.ceil(nb_train_samples / batch_size),
                                       epochs=nb_epoch,
                                       verbose=verbose,
                                       callbacks=callbacks + extra_callbacks,
                                       validation_data=generator(x_valid, y_valid, batch_size=batch_size,
                                                                 **generator_options),
                                       validation_steps=np.ceil(nb_val_samples / batch_size))

        except KeyboardInterrupt:
            return

        if return_best_model:
            self.load_all_param_values('/tmp/best_{0}.h5'.format(rn))

        self.history = self.network.history

        print('Model trained for {0} epochs. Total time: {1:.3f}s'.format(len(self.history.epoch),
                                                                          time.time() - start_time))

        return x_valid, y_valid

    def fit_generator_from_file(self, file, nb_samples, generator=None, nb_epoch=1, batch_size=64, shuffle=True,
                                validate=.0, patience=10, return_best_model=True, verbose=1, extra_callbacks=None,
                                reduce_factor=.5, **generator_options):

        # Check arguments
        if generator is None:
            generator = self.generator
        if extra_callbacks is None:
            extra_callbacks = []
        assert (validate >= 0)
        assert nb_epoch > 0

        # Callbacks
        es = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=reduce_factor,
                                      patience=patience / 2, min_lr=0.001, verbose=1)
        rn = np.random.random()
        checkpoint = ModelCheckpoint('/tmp/best_{0}.h5'.format(rn), monitor='val_loss', verbose=1, mode='min',
                                     save_best_only=True, save_weights_only=True)

        if K.backend() == "tensorflow":
            tb = TensorBoard()
            callbacks = [es, reduce_lr, checkpoint, tb]
        else:
            callbacks = [es, reduce_lr, checkpoint]

        nb_train_samples = int(nb_samples * (1 - validate))
        nb_val_samples = nb_samples - nb_train_samples

        index_array = np.arange(nb_samples, dtype=np.int32)
        if shuffle == 'shuffle':
            np.random.shuffle(index_array)

        start_time = time.time()
        try:
            self.network.fit_generator(generator=generator(file, batch_size=batch_size, **generator_options),
                                       steps_per_epoch=np.ceil(nb_train_samples / batch_size),
                                       epochs=nb_epoch,
                                       verbose=verbose,
                                       callbacks=callbacks + extra_callbacks,
                                       validation_data=generator(file, batch_size=batch_size, **generator_options),
                                       validation_steps=np.ceil(nb_val_samples / batch_size))

        except KeyboardInterrupt:
            return

        if return_best_model:
            self.load_all_param_values('/tmp/best_{0}.h5'.format(rn))

        self.history = self.network.history

        print('Model trained for {0} epochs. Total time: {1:.3f}s'.format(len(self.history.epoch),
                                                                          time.time() - start_time))

        return

    @staticmethod
    def _check_and_split_data(data, check_var=None, test_size=.0, stratify=None):
        return train_test_split(data, test_size=test_size, stratify=stratify, random_state=5)
        # Assert inputs and outputs match the model specs
        # if isinstance(data, np.ndarray):
        #     # if isinstance(check_var, list):
        #     #     raise ValueError('Number of inputs does not match model inputs.')
        #     return train_test_split(data, test_size=test_size, stratify=stratify, random_state=5)
        # elif isinstance(data, list):
        #     # if isinstance(check_var, list):
        #     #     assert (len(data) == len(check_var)), 'Number of inputs does not match model inputs.'
        #     # else:
        #     #     assert (len(data) == check_var.shape[0]), 'Number of inputs does not match model inputs.'
        #     train, test = zip(
        #         *[train_test_split(d, test_size=test_size, stratify=stratify, random_state=5) for d in data])
        #     return list(train), list(test)
        # elif isinstance(data, dict):
        #     raise NotImplementedError('Not implemented dictionary splitting yet. Please submit as list')
        # else:
        #     raise ValueError('Input data has unrecognizable format. Expecting either numpy.ndarray, list or dictionary')

    def fit(self, x_data, y_data,
            nb_inputs=1,
            nb_outputs=1,
            epochs=1,
            batch_size=64,
            shuffle=True,
            validate=.2,
            patience=10,
            return_best_model=True,
            verbose=1,
            extra_callbacks=None,
            reduce_factor=.5):

        # Check arguments
        if extra_callbacks is None:
            extra_callbacks = []
        assert (validate >= 0)
        assert epochs > 0

        if self.classification:
            stratify = y_data  # TODO: here you should select with which part to stratify
        else:
            stratify = None

        if nb_inputs == 1:
            x_train, x_valid = self._check_and_split_data(x_data, self.input, validate, stratify)
        else:
            stratify = None
            x_train = [[] for _ in x_data]
            x_valid = [[] for _ in x_data]
            for i, x_d in enumerate(x_data):
                x_train[i], x_valid[i] = self._check_and_split_data(x_d, self.network.inputs[i],
                                                                    validate, stratify)
        if nb_outputs == 1:
            y_train, y_valid = self._check_and_split_data(y_data, self.output, validate, stratify)
        else:
            stratify = None
            y_train = [[] for _ in y_data]
            y_valid = [[] for _ in y_data]
            for i, y_d in enumerate(y_data):
                y_train[i], y_valid[i] = self._check_and_split_data(y_d, self.output[i], validate, stratify)

        # Callbacks
        es = EarlyStopping(monitor='val_loss',
                           min_delta=0.0001,
                           patience=patience,
                           verbose=1,
                           mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=reduce_factor,
                                      patience=5,
                                      min_lr=0.001,
                                      verbose=1)
        rn = np.random.random()
        checkpoint = ModelCheckpoint('/tmp/best_{0}.h5'.format(rn),
                                     monitor='val_loss',
                                     verbose=1,
                                     mode='min',
                                     save_best_only=True,
                                     save_weights_only=True)

        callbacks = [es, reduce_lr, checkpoint]
        if K.backend() == "tensorflow":
            tb = TensorBoard()
            callbacks.append(tb)

        if self.classification:
            # callbacks.append(ClassificationMetrics((x_valid, y_valid)))
            callbacks.append(ClassificationMetrics())

        start_time = time.time()
        try:
            self.network.fit(x_train, y_train,
                             validation_data=(x_valid, y_valid),
                             shuffle=shuffle,
                             epochs=epochs,
                             batch_size=batch_size,
                             callbacks=callbacks + extra_callbacks,
                             verbose=verbose)
        except KeyboardInterrupt:
            pass

        if return_best_model:
            try:
                self.load_all_param_values('/tmp/best_{0}.h5'.format(rn))
            except:
                print('Unable to load best parameters, saving current model.')

        self.history = self.network.history

        print('Model trained for {0} epochs. Total time: {1:.3f}s'.format(len(self.history.epoch),
                                                                          time.time() - start_time))

        return x_valid, y_valid

    def k_fold(self, x_data, y_data, epochs=1, num_folds=10, stratify=False):
        raise NotImplementedError

    def score(self, x_data, y_data, **options):
        return self.network.evaluate(x_data, y_data, verbose=options.pop('verbose', 0), **options)

    def score_generator(self, generator=None, batch_size=1, **options):
        if generator is None:
            generator = self.generator

        if 'x_data' in options:
            if self.nb_inputs == 1:
                nb_samples = len(options.x_data)
            else:
                nb_samples = len(options.x_data[0])
        else:
            try:
                nb_samples = options.pop('nb_samples')
            except IndexError:
                raise IndexError('Must give x_data or nb_samples argumnet')

        return self.network.evaluate_generator(generator=generator(batch_size=batch_size, shuffle=False, **options),
                                               steps=np.ceil(nb_samples / batch_size),
                                               workers=1)

    def predict_proba(self, x_data):
        return self.network.predict_proba(x_data)

    def predict_classes(self, x_data):
        return self.network.predict_classes(x_data)

    def predict(self, x_data):
        return self.network.predict(x_data)

    def predict_generator(self, x_data, generator=None, batch_size=1):
        if generator is None:
            generator = self.generator

        if self.nb_inputs == 1:
            nb_train_samples = len(x_data)
        else:
            nb_train_samples = len(x_data[0])

        return self.network.predict_generator(generator=generator(x_data, batch_size=batch_size, shuffle=False),
                                              steps=np.ceil(nb_train_samples / batch_size))

    def display_network_info(self):

        print("Neural Network has {0} trainable parameters".format(self.n_params))

        layers = self.get_all_layers()

        ids = list(range(len(layers)))

        names = [layer.name for layer in layers]

        shapes = ['x'.join(map(str, layer.output_shape[1:])) for layer in layers]
        # TODO: maybe show weights shape also

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
        return [x for x in self.get_all_layers() if x.name.find('Conv') == 0]

    def get_last_conv_layer(self):
        return [x for x in self.get_all_layers() if x.name.find('Conv') == 0][-1]

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
        handle.ftype = 'history'
        handle.epochs = len(self.history.epoch)

        filename = 'models/' + handle
        if not os.path.exists('/'.join(filename.split('/')[:-1])):
            os.makedirs('/'.join(filename.split('/')[:-1]))

        np.savez_compressed(filename, **self.history.history)

        print('History saved to: ' + filename)


def inspect_inputs(i, node, fn):
    print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs],
          end='')


def inspect_outputs(i, node, fn):
    print(" output(s) value(s):", [output[0] for output in fn.outputs])

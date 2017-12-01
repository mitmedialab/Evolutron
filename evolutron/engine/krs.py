# -*- coding: utf-8 -*-
from __future__ import division, print_function

import json
import os
import time
import warnings
from collections import OrderedDict, defaultdict

import h5py
import keras
import keras.backend as K
import keras.optimizers as opt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.engine import topology
from keras.layers import deserialize_keras_object
from sklearn.model_selection import train_test_split
from tabulate import tabulate

if K.backend() == 'theano':
    from theano.compile.nanguardmode import NanGuardMode
    from theano.compile.monitormode import MonitorMode


def load_model(filepath, custom_objects=None, compile=True):
    """Loads an Evolutron model saved via Model.save().
    # Arguments
        filepath: String, path to the saved model.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.
        compile: Boolean, whether to compile the model
            after loading.
    # Returns
        An Evolutron model instance. If an optimizer was found
        as part of the saved model, the model is already
        compiled. Otherwise, the model is uncompiled and
        a warning will be displayed. When `compile` is set
        to False, the compilation is omitted without any
        warning.
    # Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    if h5py is None:
        raise ImportError('`load_model` requires h5py.')

    if not custom_objects:
        custom_objects = {}

    def convert_custom_objects(obj):
        """Handles custom object lookup.
        # Arguments
            obj: object, dict, or list.
        # Returns
            The same structure, where occurences
                of a custom object name have been replaced
                with the custom object.
        """
        if isinstance(obj, list):
            deserialized = []
            for value in obj:
                if value in custom_objects:
                    deserialized.append(custom_objects[value])
                else:
                    deserialized.append(value)
            return deserialized
        if isinstance(obj, dict):
            deserialized = {}
            for key, value in obj.items():
                deserialized[key] = []
                if isinstance(value, list):
                    for element in value:
                        if element in custom_objects:
                            deserialized[key].append(custom_objects[element])
                        else:
                            deserialized[key].append(element)
                elif value in custom_objects:
                    deserialized[key] = custom_objects[value]
                else:
                    deserialized[key] = value
            return deserialized
        if obj in custom_objects:
            return custom_objects[obj]
        return obj

    f = h5py.File(filepath, mode='r')

    # instantiate model
    model_config = f.attrs.get('model_config')
    if model_config is None:
        raise ValueError('No model found in config file.')
    model_config = json.loads(model_config.decode('utf-8'))

    globs = globals()  # All layers.
    globs['Model'] = Model
    model = deserialize_keras_object(model_config,
                                     module_objects=globs,
                                     custom_objects=custom_objects,
                                     printable_module_name='layer')
    # set weights
    topology.load_weights_from_hdf5_group(f['model_weights'], model.layers)

    # Early return if compilation is not required.
    if not compile:
        f.close()
        return model

    # instantiate optimizer
    training_config = f.attrs.get('training_config')
    if training_config is None:
        warnings.warn('No training configuration found in save file: '
                      'the model was *not* compiled. Compile it manually.')
        f.close()
        return model
    training_config = json.loads(training_config.decode('utf-8'))
    optimizer_config = training_config['optimizer_config']
    optimizer = opt.deserialize(optimizer_config,
                                custom_objects=custom_objects)

    # Recover loss functions and metrics.
    loss = convert_custom_objects(training_config['loss'])
    metrics = convert_custom_objects(training_config['metrics'])
    sample_weight_mode = training_config['sample_weight_mode']
    loss_weights = training_config['loss_weights']

    # Compile model.
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics,
                  loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode)

    # Set optimizer weights.
    if 'optimizer_weights' in f:
        # Build train function (to get weight updates).
        model._make_train_function()
        optimizer_weights_group = f['optimizer_weights']
        optimizer_weight_names = [n.decode('utf8') for n in optimizer_weights_group.attrs['weight_names']]
        optimizer_weight_values = [optimizer_weights_group[n] for n in optimizer_weight_names]
        model.optimizer.set_weights(optimizer_weight_values)
    f.close()
    return model


class Model(keras.models.Model):
    def __init__(self, inputs, outputs, name=None,
                 classification=False,
                 verbose=False,
                 generator=None,
                 nb_inputs=1,
                 nb_outputs=1):

        self.verbose = verbose
        self.generator = generator
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs

        self.classification = classification

        self.history = None
        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []

        self.fold_train_losses = None
        self.fold_val_losses = None
        self.k_fold_history = defaultdict(list)

        self.history_list = []

        super(Model, self).__init__(inputs=inputs, outputs=outputs, name=name)

    def compile(self, optimizer, loss, metrics=None, loss_weights=None,
                sample_weight_mode=None, **options):

        if isinstance(optimizer, str):
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
            optimizer = opts[optimizer]

        mode = options.get('mode', None)
        if K.backend() == 'theano' and mode:
            modes = {'NaNGuardMode': NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
                     'MonitorMode': MonitorMode(pre_func=inspect_inputs, post_func=inspect_outputs),
                     }
            mode = modes[mode]

            super(Model, self).compile(loss=loss,
                                       loss_weights=loss_weights,
                                       optimizer=optimizer,
                                       metrics=metrics,
                                       sample_weight_mode=sample_weight_mode,
                                       mode=mode)
        else:
            super(Model, self).compile(loss=loss,
                                       loss_weights=loss_weights,
                                       optimizer=optimizer,
                                       sample_weight_mode=sample_weight_mode,
                                       metrics=metrics)

    def fit_generator(self, x_data, y_data,
                      generator=None,
                      epochs=1,
                      verbose=1,
                      callbacks=None,
                      validation_data=None,
                      validation_split=.0,
                      validation_steps=None,
                      class_weight=None,
                      max_q_size=10,
                      workers=1,
                      pickle_safe=False,
                      initial_epoch=0,
                      batch_size=64, shuffle=True,
                      return_best_model=True,
                      monitor='val_loss',
                      **generator_options):

        # Check arguments
        if generator is None:
            generator = self.generator

        assert (validation_split >= 0)
        assert epochs > 0

        if validation_data is None:
            if self.classification:
                stratify = y_data  # TODO: here you should select with which part to stratify
            else:
                stratify = None

            if self.nb_inputs == 1:
                x_train, x_valid = train_test_split(x_data, test_size=validation_split, stratify=stratify,
                                                    random_state=5)
            else:
                stratify = None
                x_train = [[] for _ in x_data]
                x_valid = [[] for _ in x_data]
                for i, x_d in enumerate(x_data):
                    x_train[i], x_valid[i] = train_test_split(x_d, test_size=validation_split, stratify=stratify,
                                                              random_state=5)
            if self.nb_outputs == 1:
                y_train, y_valid = train_test_split(y_data, test_size=validation_split, stratify=stratify,
                                                    random_state=5)
            else:
                stratify = None
                y_train = [[] for _ in y_data]
                y_valid = [[] for _ in y_data]
                for i, y_d in enumerate(y_data):
                    y_train[i], y_valid[i] = train_test_split(y_d, test_size=validation_split, stratify=stratify,
                                                              random_state=5)
        else:
            x_train = x_data
            y_train = y_data
            x_valid = validation_data[0]
            y_valid = validation_data[1]

        if self.nb_inputs == 1:
            nb_train_samples = len(x_train)
            nb_val_samples = len(x_valid)
        else:
            nb_train_samples = len(x_train[0])
            nb_val_samples = len(x_valid[0])

        if return_best_model:
            rn = np.random.random()
            checkpoint = ModelCheckpoint('/tmp/best_{0}.h5'.format(rn),
                                         monitor=monitor,
                                         verbose=1,
                                         mode='min',
                                         save_best_only=True,
                                         save_weights_only=True)
            callbacks.append(checkpoint)

        for cb in callbacks:
            cb.validation_data = (x_valid, y_valid)

        start_time = time.time()
        try:
            super(Model, self).fit_generator(generator=generator(x_train, y_train, batch_size=batch_size,
                                                                 shuffle=shuffle, **generator_options),
                                             steps_per_epoch=np.ceil(nb_train_samples / batch_size),
                                             epochs=epochs,
                                             verbose=verbose,
                                             callbacks=callbacks,
                                             validation_data=generator(x_valid, y_valid, batch_size=batch_size,
                                                                       **generator_options),
                                             validation_steps=np.ceil(nb_val_samples / batch_size),
                                             workers=workers,
                                             max_q_size=max_q_size,
                                             pickle_safe=pickle_safe,
                                             initial_epoch=initial_epoch)

        except KeyboardInterrupt:
            return

        if return_best_model:
            try:
                self.load_all_param_values('/tmp/best_{0}.h5'.format(rn))
            except:
                print('Unable to load best parameters, saving current model.')

        self.history_list.append(self.history)

        print('Model trained for {0} epochs. Total time: {1:.3f}s'.format(len(self.history.epoch),
                                                                          time.time() - start_time))

        return x_valid, y_valid

    def fit_generator_from_file(self, nb_samples=None,
                                generator=None,
                                steps_per_epoch=None,
                                epochs=1,
                                verbose=1,
                                callbacks=None,
                                validation_data=None,
                                validation_steps=None,
                                validation_split=.0,
                                class_weight=None,
                                max_q_size=10,
                                workers=1,
                                use_multiprocessing=False,
                                initial_epoch=0,
                                batch_size=64,
                                shuffle=True,
                                return_best_model=True,
                                monitor='val_loss',
                                **generator_options):

        # Check arguments
        if generator is None:
            generator = self.generator

        assert (validation_split >= 0)
        assert epochs > 0

        if steps_per_epoch is None:
            nb_train_samples = int(nb_samples * (1 - validation_split))
            steps_per_epoch = np.ceil(nb_train_samples / batch_size)
        if validation_steps is None:
            nb_val_samples = nb_samples - nb_train_samples
            validation_steps = np.ceil(nb_val_samples / batch_size)

        if callable(generator):
            generator = generator(batch_size=batch_size, **generator_options)

        if validation_data is None:
            validation_data = generator

        """
        index_array = np.arange(nb_samples, dtype=np.int32)
        if shuffle == 'shuffle':
            np.random.shuffle(index_array)
        """

        if return_best_model:
            rn = np.random.random()
            checkpoint = ModelCheckpoint('/tmp/best_{0}.h5'.format(rn),
                                         monitor=monitor,
                                         verbose=1,
                                         mode='min',
                                         save_best_only=True,
                                         save_weights_only=True)
            callbacks.append(checkpoint)

        for cb in callbacks:
            cb.validation_data = validation_data
            cb.validation_steps = validation_steps

        start_time = time.time()
        try:
            super(Model, self).fit_generator(generator=generator,
                                             steps_per_epoch=steps_per_epoch,
                                             epochs=epochs,
                                             verbose=verbose,
                                             callbacks=callbacks,
                                             validation_data=validation_data,
                                             validation_steps=validation_steps,
                                             workers=workers,
                                             max_q_size=max_q_size,
                                             initial_epoch=initial_epoch,
                                             class_weight=class_weight,
                                             use_multiprocessing=use_multiprocessing)

        except KeyboardInterrupt:
            return

        if return_best_model:
            try:
                self.load_all_param_values('/tmp/best_{0}.h5'.format(rn))
            except:
                print('Unable to load best parameters, saving current model.')

        self.history_list.append(self.history)

        print('Model trained for {0} epochs. Total time: {1:.3f}s'.format(len(self.history.epoch),
                                                                          time.time() - start_time))

        return

    def fit(self, x=None, y=None,
            batch_size=32,
            epochs=1,
            initial_epoch=0,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            return_best_model=True,
            reduce_factor=.5,
            **kwargs):

        # Check arguments
        assert (validation_split >= 0)
        assert epochs > 0

        if validation_data:
            x_train, y_train = x, y
            x_valid, y_valid = validation_data

        elif validation_split > 0:

            if self.classification:
                stratify = y
            else:
                stratify = None

            if self.nb_inputs == 1:
                x_train, x_valid = train_test_split(x, test_size=validation_split,
                                                    stratify=stratify,
                                                    shuffle=shuffle,
                                                    random_state=5)
            else:
                stratify = None
                x_train = [[] for _ in x]
                x_valid = [[] for _ in x]
                for i, x_d in enumerate(x):
                    x_train[i], x_valid[i] = train_test_split(x_d, test_size=validation_split, stratify=stratify,
                                                              random_state=5)
            if self.nb_outputs == 1:
                y_train, y_valid = train_test_split(y, test_size=validation_split, stratify=stratify, random_state=5)
            else:
                stratify = None
                y_train = [[] for _ in y]
                y_valid = [[] for _ in y]
                for i, y_d in enumerate(y):
                    y_train[i], y_valid[i] = train_test_split(y_d, test_size=validation_split, stratify=stratify,
                                                              random_state=5)

        else:
            x_train, y_train = x, y
            x_valid, y_valid = [], []

        if return_best_model:
            rn = np.random.random()
            checkpoint = ModelCheckpoint('/tmp/best_{0}.h5'.format(rn),
                                         monitor='val_loss',
                                         verbose=1,
                                         mode='min',
                                         save_best_only=True,
                                         save_weights_only=True)
            # TODO: this is multiply defined, define only once per fit
            callbacks.append(checkpoint)

        start_time = time.time()
        try:
            super(Model, self).fit(x=x_train, y=y_train,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   initial_epoch=initial_epoch,
                                   verbose=verbose,
                                   callbacks=callbacks,
                                   validation_data=(x_valid, y_valid),
                                   shuffle=True,
                                   class_weight=None,
                                   sample_weight=None)
        except KeyboardInterrupt:
            pass

        if return_best_model:
            try:
                self.load_all_param_values('/tmp/best_{0}.h5'.format(rn))
            except:
                print('Unable to load best parameters, saving current model.')

        self.history_list.append(self.history)

        print('Model trained for {0} epochs. Total time: {1:.3f}s'.format(len(self.history.epoch),
                                                                          time.time() - start_time))

        return x_valid, y_valid

    def evaluate_generator(self, generator=None, steps=None, batch_size=1, **options):
        if generator is None:
            generator = self.generator(batch_size=batch_size, shuffle=False, **options)

        if steps is None:
            if 'x_data' in options and self.nb_inputs == 1:
                nb_samples = len(options['x_data'])
            elif 'x_data' in options:
                nb_samples = len(options['x_data'][0])
            elif 'nb_samples' in options:
                nb_samples = options.pop('nb_samples')
            else:
                raise KeyError('Must give x_data or nb_samples argument')

            steps = np.ceil(nb_samples / batch_size)

        return super(Model, self).evaluate_generator(generator=generator,
                                                     steps=steps)

    def predict_generator(self, x_data=None, generator=None, steps=None, batch_size=1, **options):
        if generator is None:
            generator = self.generator(x_data, batch_size=batch_size, shuffle=False)

            if self.nb_inputs == 1:
                nb_train_samples = len(x_data)
            else:
                nb_train_samples = len(x_data[0])
            steps = np.ceil(nb_train_samples / batch_size)
        else:
            assert steps is not None

        return super(Model, self).predict_generator(generator=generator,
                                                    steps=steps,
                                                    max_q_size=10,
                                                    workers=1,
                                                    pickle_safe=False)

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
        return self.layers

    def get_all_params(self, trainable=True):
        if trainable:
            return self.weights
        else:
            return self.non_trainable_weights

    @property
    def n_params(self):
        return self.count_params()

    def get_all_param_values(self):
        return self.get_weights()

    def set_all_param_values(self, weights):
        try:
            self.set_weights(weights)
        except Exception:
            msg = 'Incorrect parameter list'
            raise ValueError(msg)

    def load_all_param_values(self, filepath):
        try:
            self.load_weights(filepath)
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

    def save(self, handle, data_dir=None, **save_args):

        handle.ftype = 'model'
        handle.epochs = len(self.history.epoch)
        filename = 'models/' + handle
        if data_dir:
            filename = os.path.join(data_dir, filename)

        if not os.path.exists('/'.join(filename.split('/')[:-1])):
            os.makedirs('/'.join(filename.split('/')[:-1]))

        super(Model, self).save(filename, **save_args)

        print('Model saved to: ' + filename)

    def save_train_history(self, handle, data_dir=None):
        handle.ftype = 'history'
        handle.epochs = len(self.history.epoch)
        filename = 'models/' + handle
        if data_dir:
            filename = os.path.join(data_dir, filename)

        if not os.path.exists('/'.join(filename.split('/')[:-1])):
            os.makedirs('/'.join(filename.split('/')[:-1]))

        np.savez_compressed(filename, **self.history.history)

        print('History saved to: ' + filename)


def inspect_inputs(i, node, fn):
    print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs],
          end='')


def inspect_outputs(i, node, fn):
    print(" output(s) value(s):", [output[0] for output in fn.outputs])

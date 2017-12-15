# -*- coding: utf-8 -*-
from __future__ import division, print_function

import json
import os
import time
import warnings
from collections import defaultdict

import h5py
import keras
import keras.backend as K
import keras.optimizers as opt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.engine import topology
from keras.layers import deserialize_keras_object
from keras.utils import print_summary

from ..tools import Handle, train_valid_split


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
                 nb_inputs=1,
                 nb_outputs=1):

        self.verbose = verbose
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs

        self.classification = classification

        self.history = None
        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []
        self.training_duration = 0.0

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

    def fit_generator(self, epochs=1, initial_epoch=0, verbose=1, callbacks=None, class_weight=None, max_queue_size=10,
                      workers=4, use_multiprocessing=True, shuffle=True, return_best_model=True, monitor='val_loss',
                      **data_arguments):

        if return_best_model:
            rn = np.random.random()
            checkpoint = ModelCheckpoint('/tmp/best_{0}.h5'.format(rn),
                                         monitor=monitor,
                                         verbose=verbose,
                                         mode='min',
                                         save_best_only=True,
                                         save_weights_only=True)
            if isinstance(callbacks, list):
                callbacks.append(checkpoint)
            else:
                callbacks = [checkpoint]

        start_time = time.time()
        super(Model, self).fit_generator(verbose=verbose,
                                         initial_epoch=initial_epoch,
                                         epochs=epochs,
                                         callbacks=callbacks,
                                         workers=workers,
                                         use_multiprocessing=use_multiprocessing,
                                         max_queue_size=max_queue_size,
                                         shuffle=shuffle,
                                         **data_arguments)
        end_time = time.time()
        print('Model trained for {0} epochs. Total time: {1:.3f}s'.format(len(self.history.epoch),
                                                                          end_time - start_time))
        self.training_duration = end_time - start_time

        if return_best_model:
            try:
                self.load_all_param_values('/tmp/best_{0}.h5'.format(rn))
            except:
                print('Unable to load best parameters, saving current model.')

        self.history_list.append(self.history)

    def fit(self, x=None, y=None, batch_size=32, epochs=1, initial_epoch=0, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True, return_best_model=True, reduce_factor=.5,
            monitor='val_loss', **fit_kwargs):

        # Check arguments
        assert (validation_split >= 0)
        assert epochs > 0

        if self.classification:
            stratify = y
        else:
            stratify = None

        if validation_data:
            x_train, y_train = x, y
        else:
            if validation_split > 0:
                x_train, y_train, x_valid, y_valid = train_valid_split(x, y, validation_split=validation_split,
                                                                       shuffle=shuffle, stratify=stratify)
                validation_data = x_valid, y_valid
            else:
                x_train, y_train = x, y
                validation_data = None

        if return_best_model:
            rn = np.random.random()
            checkpoint = ModelCheckpoint('/tmp/best_{0}.h5'.format(rn),
                                         monitor=monitor,
                                         verbose=verbose,
                                         mode='min',
                                         save_best_only=True,
                                         save_weights_only=True)
            if isinstance(callbacks, list):
                callbacks.append(checkpoint)
            else:
                callbacks = [checkpoint]

        start_time = time.time()
        super(Model, self).fit(x=x_train, y=y_train,
                               batch_size=batch_size,
                               epochs=epochs,
                               initial_epoch=initial_epoch,
                               verbose=verbose,
                               callbacks=callbacks,
                               validation_data=validation_data,
                               shuffle=shuffle,
                               **fit_kwargs)
        end_time = time.time()
        print('Model trained for {0} epochs. Total time: {1:.3f}s'.format(len(self.history.epoch),
                                                                          end_time - start_time))
        self.training_duration = end_time - start_time
        if return_best_model:
            try:
                self.load_all_param_values('/tmp/best_{0}.h5'.format(rn))
            except Exception as e:
                print('Unable to load best parameters, saving current model.')
                print(e)

        self.history_list.append(self.history)

        return x_valid, y_valid

    # TODO: implement predict/evaluate generator

    def display_network_info(self, line_length=100):
        print_summary(self, line_length=line_length)

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

        if isinstance(handle, Handle):
            handle.ftype = 'model'
            handle.epochs = len(self.history.epoch)
        else:
            handle += '.model'

        filename = 'models/' + handle
        if data_dir:
            filename = os.path.join(data_dir, filename)

        if not os.path.exists('/'.join(filename.split('/')[:-1])):
            os.makedirs('/'.join(filename.split('/')[:-1]))

        super(Model, self).save(filename, **save_args)

        print('Model saved to: ' + filename)

    def save_architecture(self, handle, data_dir=None, filetype='json'):

        if isinstance(handle, Handle):
            handle.ftype = 'arch'
            handle.epochs = len(self.history.epoch)
        else:
            handle += '.arch'

        filename = 'models/' + handle
        if data_dir:
            filename = os.path.join(data_dir, filename)

        if not os.path.exists('/'.join(filename.split('/')[:-1])):
            os.makedirs('/'.join(filename.split('/')[:-1]))

        if filetype == 'json':
            with open(filename, 'w') as f:
                f.write(super(Model, self).to_json())
        elif filetype == 'yaml':
            with open(filename, 'w') as f:
                f.write(super(Model, self).to_yaml())
        else:
            raise ValueError('Invalid file type for saving.')

        print('Architecture saved to: ' + filename)

    def save_train_history(self, handle, data_dir=None):

        if isinstance(handle, Handle):
            handle.ftype = 'history'
            handle.epochs = len(self.history.epoch)
        else:
            handle += '.history'
        filename = 'models/' + handle
        if data_dir:
            filename = os.path.join(data_dir, filename)

        if not os.path.exists('/'.join(filename.split('/')[:-1])):
            os.makedirs('/'.join(filename.split('/')[:-1]))

        np.savez_compressed(filename, training_duration=self.training_duration, **self.history.history)

        print('History saved to: ' + filename)


def inspect_inputs(i, node, fn):
    print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs],
          end='')


def inspect_outputs(i, node, fn):
    print(" output(s) value(s):", [output[0] for output in fn.outputs])

"""
    Implementation of a Convolutional Autoencoder
    in Keras.

    Thrasyvoulos Karydis
    (c) Massachusetts Institute of Technology 2016

    This work may be reproduced, modified, distributed, performed, and
    displayed for any purpose, but must acknowledge the mods
    project. Copyright is retained and must be preserved. The work is
    provided as is; no warranty is provided, and users accept all
    liability.
"""
from keras.layers import Input, Convolution1D, MaxPooling1D, Dense, Flatten, Reshape, UpSampling1D, Dropout
from keras.models import Model, load_model
from keras.objectives import categorical_crossentropy, mean_squared_error
from keras.metrics import categorical_accuracy

import keras.backend as K


class DeepCoDER(Model):
    def __init__(self, input, output, name=None):
        super(DeepCoDER, self).__init__(input, output, name)

        convs = [l for l in self.layers if l.name.find('Conv') == 0]

        deconvs = [l for l in self.layers if l.name.find('Deconv') == 0]

        deconvs = deconvs[::-1]

        for c, d in zip(convs, deconvs):
            d.W = c.W.transpose((0, 1, 3, 2))
            d.b = c.b

    @classmethod
    def from_options(cls, aa_length, n_filters, filter_length, n_conv_layers=1, n_fc_layers=1):
        args = cls._build_network(aa_length, n_conv_layers, n_fc_layers, n_filters, filter_length)

        args['name'] = 'dtkrCoDER'

        return cls(**args)

    @classmethod
    def from_saved_model(cls, filepath):
        cls.__dict__ = load_model(filepath)
        cls.__class__ = DeepCoDER
        return cls

    def save(self, filepath, overwrite=True):
        self.__class__.__name__ = 'Model'
        super(DeepCoDER, self).save(filepath, overwrite=overwrite)

    @staticmethod
    def _build_network(aa_length, n_conv_layers, n_fc_layers, n_filters, filter_length):
        inp = Input(shape=(aa_length, 20), name='aa_seq')  # Assuming tf dimension ordering

        convs = [Convolution1D(n_filters, filter_length,
                               init='glorot_uniform',  # change that for gaussian
                               activation='linear',
                               border_mode='same',
                               name='Conv1')(inp)]

        for c in range(2, n_conv_layers + 1):
            convs.append(Convolution1D(n_filters, filter_length,
                                       init='glorot_uniform',  # change that for gaussian
                                       activation='linear',
                                       border_mode='same',
                                       name='Conv' + str(c))(convs[-1]))  # maybe add L1 regularizer

        max_pool = MaxPooling1D(pool_length=aa_length)(convs[-1])

        flat = Flatten()(max_pool)

        dense = Dense(n_filters, init='glorot_uniform', activation='linear')(Dropout(.5)(flat))

        encoded = Dense(n_filters, init='glorot_uniform', activation='sigmoid')(Dropout(.5)(dense))

        dedense = Dense(n_filters, init='glorot_uniform', activation='linear')(encoded)

        dedense = Dense(n_filters, init='glorot_uniform', activation='linear')(dedense)

        unflat = Reshape(max_pool._keras_shape[1:])(dedense)

        deconvs = [UpSampling1D(length=aa_length, name='Unpooling')(unflat)]

        for c in range(n_conv_layers, 1, -1):
            deconvs.append(Convolution1D(n_filters, filter_length,
                                         init='glorot_uniform',  # change that for gaussian
                                         activation='linear',
                                         border_mode='same',
                                         name='Deconv' + str(c))(deconvs[-1]))  # maybe add L1 regularizer

        decoded = Convolution1D(20, filter_length, activation='sigmoid', border_mode='same', name='Deconv1')(
            deconvs[-1])

        return {'input': inp, 'output': decoded}

    @property
    def _loss_function(self):
        return 'mse'

    @staticmethod
    def mean_cat_acc(inp, decoded):
        y_true = K.reshape(inp, (-1, 20))
        y_pred = K.reshape(decoded, (-1, 20))
        cat_acc = categorical_accuracy(y_true, y_pred)
        return cat_acc


class DeepCoDERwCross(Model):
    def __init__(self, input, output, name=None):
        super(DeepCoDERwCross, self).__init__(input, output, name)

        convs = [l for l in self.layers if l.name.find('Conv') == 0]

        deconvs = [l for l in self.layers if l.name.find('Deconv') == 0]

        deconvs = deconvs[::-1]

        for c, d in zip(convs, deconvs):
            d.W = K.permute_dimensions(c.W, (0, 1, 3, 2))
            d.b = c.b

    @classmethod
    def from_options(cls, aa_length, n_filters, filter_length, n_conv_layers=1, n_fc_layers=1):
        args = cls._build_network(aa_length, n_conv_layers, n_fc_layers, n_filters, filter_length)

        args['name'] = 'krCoDERwCross'

        return cls(**args)

    @classmethod
    def from_saved_model(cls, filepath):
        cls.__dict__ = load_model(filepath)
        cls.__class__ = DeepCoDERwCross
        return cls

    def save(self, filepath, overwrite=True):
        self.__class__.__name__ = 'Model'
        super(DeepCoDERwCross, self).save(filepath, overwrite=overwrite)

    @staticmethod
    def _build_network(aa_length, n_conv_layers, n_fc_layers, n_filters, filter_length):
        inp = Input(shape=(aa_length, 20), name='aa_seq')  # Assuming tf dimension ordering

        convs = [Convolution1D(n_filters, filter_length,
                               init='glorot_normal',  # change that for gaussian
                               activation='sigmoid',
                               border_mode='same',
                               name='Conv1')(inp)]

        for c in range(2, n_conv_layers + 1):
            convs.append(Convolution1D(n_filters, filter_length,
                                       init='glorot_normal',  # change that for gaussian
                                       activation='sigmoid',
                                       border_mode='same',
                                       name='Conv' + str(c))(convs[-1]))  # maybe add L1 regularizer

        max_pool = MaxPooling1D(pool_length=aa_length)(convs[-1])

        flat = Flatten()(max_pool)

        dense = Dense(n_filters, init='glorot_normal', activation='sigmoid')(Dropout(.5)(flat))

        encoded = Dense(2 * n_filters, init='glorot_normal', activation='sigmoid')(Dropout(.5)(dense))

        dedense = Dense(n_filters, init='glorot_normal', activation='sigmoid')(encoded)

        dedense = Dense(n_filters, init='glorot_normal', activation='sigmoid')(dedense)

        unflat = Reshape(max_pool._keras_shape[1:])(dedense)

        deconvs = [UpSampling1D(length=aa_length, name='Unpooling')(unflat)]

        for c in range(n_conv_layers, 1, -1):
            deconvs.append(Convolution1D(n_filters, filter_length,
                                         init='glorot_normal',  # change that for gaussian
                                         activation='sigmoid',
                                         border_mode='same',
                                         name='Deconv' + str(c))(deconvs[-1]))  # maybe add L1 regularizer

        decoded = Convolution1D(20, filter_length, activation='sigmoid', border_mode='same', name='Deconv1')(
            deconvs[-1])

        return {'input': inp, 'output': decoded}

    @staticmethod
    def _loss_function(inp, decoded):
        # y_true = K.reshape(inp, (-1, 20))
        # y_pred = K.reshape(decoded, (-1, 20))
        # loss = K.mean(categorical_crossentropy(y_true, y_pred))
        loss = mean_squared_error(y_true=inp, y_pred=decoded)
        return loss

    @staticmethod
    def mean_cat_acc(inp, decoded):
        y_true = K.reshape(inp, (-1, 20))
        y_pred = K.reshape(decoded, (-1, 20))
        cat_acc = categorical_accuracy(y_true, y_pred)
        return cat_acc


class swDeepCoDER(Model):
    def __init__(self, input, output, name=None):
        super(swDeepCoDER, self).__init__(input, output, name)

        # convs = [l for l in self.layers if l.name.find('Conv') == 0]
        #
        # deconvs = [l for l in self.layers if l.name.find('Deconv') == 0]
        #
        # deconvs = deconvs[::-1]
        #
        # for c, d in zip(convs, deconvs):
        #     d.W = K.permute_dimensions(c.W, (0, 1, 3, 2))
        #     d.b = c.b

    @classmethod
    def from_options(cls, aa_length, n_filters, filter_length, n_conv_layers=1, n_fc_layers=1):
        args = cls._build_network(aa_length, n_conv_layers, n_fc_layers, n_filters, filter_length)

        args['name'] = 'krswDeepCoDER'

        return cls(**args)

    @classmethod
    def from_saved_model(cls, filepath):
        cls.__dict__ = load_model(filepath)
        cls.__class__ = DeepCoDERwCross
        return cls

    def save(self, filepath, overwrite=True):
        self.__class__.__name__ = 'Model'
        super(swDeepCoDER, self).save(filepath, overwrite=overwrite)

    @staticmethod
    def _build_network(aa_length, n_conv_layers, n_fc_layers, n_filters, filter_length):
        inp = Input(shape=(aa_length, 20), name='aa_seq')  # Assuming tf dimension ordering

        convs = [Convolution1D(n_filters, filter_length,
                               init='glorot_normal',  # change that for gaussian
                               activation='relu',
                               border_mode='same',
                               name='Conv1')(inp)]

        for c in range(2, n_conv_layers + 1):
            convs.append(Convolution1D(n_filters, filter_length,
                                       init='glorot_normal',  # change that for gaussian
                                       activation='relu',
                                       border_mode='same',
                                       name='Conv' + str(c))(convs[-1]))  # maybe add L1 regularizer

        max_pool = MaxPooling1D(pool_length=aa_length)(convs[-1])

        flat = Flatten()(max_pool)

        encoded = Dense(n_filters, init='glorot_normal', activation='sigmoid')(flat)

        dedense = Dense(n_filters, init='glorot_normal', activation='linear')(encoded)

        unflat = Reshape(max_pool._keras_shape[1:])(dedense)

        deconvs = [UpSampling1D(length=aa_length, name='Unpooling')(unflat)]

        for c in range(n_conv_layers, 1, -1):
            deconvs.append(Convolution1D(n_filters, filter_length,
                                         init='glorot_normal',  # change that for gaussian
                                         activation='relu',
                                         border_mode='same',
                                         name='Deconv' + str(c))(deconvs[-1]))  # maybe add L1 regularizer

        decoded = Convolution1D(20, filter_length, activation='sigmoid', border_mode='same', name='Deconv1')(
            deconvs[-1])

        return {'input': inp, 'output': decoded}

    @staticmethod
    def _loss_function(inp, decoded):
        # y_true = K.reshape(inp, (-1, 20))
        # y_pred = K.reshape(decoded, (-1, 20))
        # loss = K.mean(categorical_crossentropy(y_true, y_pred))
        loss = mean_squared_error(y_true=inp, y_pred=decoded)
        return loss

    @staticmethod
    def mean_cat_acc(inp, decoded):
        y_true = K.reshape(inp, (-1, 20))
        y_pred = K.reshape(decoded, (-1, 20))
        cat_acc = categorical_accuracy(y_true, y_pred)
        return cat_acc

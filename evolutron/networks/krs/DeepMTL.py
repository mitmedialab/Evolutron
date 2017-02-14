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
import keras.backend as K
from keras.layers import Input
from keras.layers import Masking
from keras.metrics import categorical_accuracy
from keras.models import Model, load_model
from keras.objectives import mean_squared_error, categorical_crossentropy

from evolutron.networks.extra_layers import Convolution1D, MaxPooling1D, Dense, Flatten, Reshape  # To implement masking
from evolutron.networks.extra_layers import Dedense, Unpooling1D, Deconvolution1D
from evolutron.networks.krs.extra_metrics import mean_cat_acc

class DeepCoDER(Model):
    def __init__(self, input, output, name=None):
        super(DeepCoDER, self).__init__(input, output, name)

        self.metrics = [self.mean_cat_acc]

        self.name = self.__class__.__name__

    @classmethod
    def from_options(cls, aa_length, n_filters, filter_length, n_conv_layers=1, n_fc_layers=1):
        args = cls._build_network(aa_length, n_conv_layers, n_fc_layers, n_filters, filter_length)

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
    def _build_network(input_shape, n_conv_layers, n_fc_layers, nb_filter, filter_length):
        assert len(input_shape) == 2, 'Unrecognizable input dimensions'
        assert K.image_dim_ordering() == 'tf', 'Theano dimension ordering not supported yet'
        assert input_shape[1] in [20, 4, 22], 'Input dimensions error, check order'

        seq_length, alphabet = input_shape

        # Input LayerRO
        inp = Input(shape=input_shape, name='aa_seq')

        mask = Masking(mask_value=0.0)(inp)

        # Convolutional Layers
        convs = [Convolution1D(nb_filter, filter_length,
                               init='glorot_uniform',
                               activation='relu',
                               border_mode='same',
                               name='Conv1')(mask)]

        for c in range(1, n_conv_layers):
            convs.append(Convolution1D(nb_filter, filter_length,
                                       init='glorot_uniform',
                                       activation='relu',
                                       border_mode='same',
                                       name='Conv{}'.format(c + 1))(convs[-1]))  # maybe add L1 regularizer

        # Max-pooling
        if seq_length:
            max_pool = MaxPooling1D(pool_length=seq_length)(convs[-1])
            flat = Flatten()(max_pool)
        else:
            # max_pool = GlobalMaxPooling1D()(convs[-1])
            # flat = max_pool
            raise NotImplementedError('Sequence length must be known at this point. Pad and use mask.')

        # Fully-Connected encoding layers
        fc_enc = [Dense(nb_filter,
                        init='glorot_uniform',
                        activation='sigmoid',
                        name='FCEnc1')(flat)]

        for d in range(1, n_fc_layers):
            fc_enc.append(Dense(nb_filter,
                                init='glorot_uniform',
                                activation='sigmoid',
                                name='FCEnc{}'.format(d + 1))(fc_enc[-1]))

        encoded = fc_enc[-1]  # To access if model for encoding needed

        # Fully-Connected decoding layers
        fc_dec = [Dedense(encoded._keras_history[0],
                          activation='linear',
                          name='FCDec{}'.format(n_fc_layers))(encoded)]

        for d in range(n_fc_layers - 2, -1, -1):
            fc_dec.append(Dedense(fc_enc[d]._keras_history[0],
                                  activation='linear',
                                  name='FCDec{}'.format(d + 1))(fc_dec[-1]))

        # Reshaping and unpooling
        if seq_length:
            unflat = Reshape(max_pool._keras_shape[1:])(fc_dec[-1])
        else:
            unflat = Reshape((1, fc_dec[-1]._keras_shape[-1]))(fc_dec[-1])

        deconvs = [Unpooling1D(max_pool._keras_history[0], name='Unpooling')(unflat)]

        # Deconvolution
        for c in range(n_conv_layers - 1, 0, -1):
            deconvs.append(Deconvolution1D(convs[c]._keras_history[0],
                                           activation='relu',
                                           name='Deconv{}'.format(c + 1))(deconvs[-1]))  # maybe add L1 regularizer

        decoded = Deconvolution1D(convs[0]._keras_history[0],
                                  apply_mask=True,
                                  activation='sigmoid',
                                  name='Deconv1')(deconvs[-1])

        return {'input': inp, 'output': decoded}

    @staticmethod
    def _loss_function(inp, decoded):
        return mean_squared_error(y_true=inp, y_pred=decoded)

    @staticmethod
    def mean_cat_acc(inp, decoded):
        return mean_cat_acc(inp, decoded)


class DeepCoFAM(Model):
    def __init__(self, input, output, name=None):
        super(DeepCoFAM, self).__init__(input, output, name)
        self.metrics = [self.mean_cat_acc]

        self.name = self.__class__.__name__

    @classmethod
    def from_options(cls, input_shape, output_dim, n_filters, filter_length, n_conv_layers=1, n_fc_layers=1):
        args = cls._build_network(input_shape, output_dim, n_conv_layers, n_fc_layers, n_filters, filter_length)

        return cls(**args)

    @classmethod
    def from_saved_model(cls, filepath):
        cls.__dict__ = load_model(filepath)
        cls.__class__ = DeepCoFAM
        return cls

    def save(self, filepath, overwrite=True):
        self.__class__.__name__ = 'Model'
        super(DeepCoFAM, self).save(filepath, overwrite=overwrite)

    @staticmethod
    def _build_network(input_shape, output_dim, n_conv_layers, n_fc_layers, nb_filter, filter_length):
        assert len(input_shape) == 2, 'Unrecognizable input dimensions'
        assert K.image_dim_ordering() == 'tf', 'Theano dimension ordering not supported yet'
        assert input_shape[1] in [20, 4, 22], 'Input dimensions error, check order'

        seq_length, alphabet = input_shape

        # Input LayerRO
        inp = Input(shape=input_shape, name='aa_seq')

        # Convolutional Layers
        convs = [Convolution1D(nb_filter, filter_length,
                               init='glorot_uniform',
                               activation='relu',
                               border_mode='same',
                               name='Conv1')(inp)]

        for c in range(1, n_conv_layers):
            convs.append(Convolution1D(nb_filter, filter_length,
                                       init='glorot_uniform',
                                       activation='relu',
                                       border_mode='same',
                                       name='Conv{}'.format(c + 1))(convs[-1]))  # maybe add L1 regularizer

        # Max-pooling
        if seq_length:
            max_pool = MaxPooling1D(pool_length=seq_length)(convs[-1])
            flat = Flatten()(max_pool)
        else:
            # max_pool = GlobalMaxPooling1D()(convs[-1])
            # flat = max_pool
            raise NotImplementedError('Sequence length must be known at this point. Pad and use mask.')

        # Fully-Connected encoding layers
        fc_enc = [Dense(nb_filter,
                        init='glorot_uniform',
                        activation='sigmoid',
                        name='FCEnc1')(flat)]

        for d in range(1, n_fc_layers):
            fc_enc.append(Dense(nb_filter,
                                init='glorot_uniform',
                                activation='sigmoid',
                                name='FCEnc{}'.format(d + 1))(fc_enc[-1]))

        encoded = fc_enc[-1]  # To access if model for encoding needed

        classifier = Dense(output_dim,
                           init='glorot_uniform',
                           activation='softmax',
                           name='Classifier')(encoded)

        return {'input': inp, 'output': classifier}

    @staticmethod
    def _loss_function(y_true, y_pred):
        loss = categorical_crossentropy(y_true=y_true, y_pred=y_pred)
        return loss

    @staticmethod
    def mean_cat_acc(y_true, y_pred):
        cat_acc = categorical_accuracy(y_pred, y_true)
        return cat_acc

"""
    Implementation of a Convolutional DNA recognition sequence classifier
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
from keras.layers import Merge
from keras.metrics import categorical_accuracy
from keras.models import Model, load_model
from keras.objectives import categorical_crossentropy

from .extra_layers import Convolution1D, MaxPooling1D, Dense, Flatten  # To implement masking


class DeepDNABind(Model):
    def __init__(self, input, output, name=None):
        super(DeepDNABind, self).__init__(input, output, name)

    @classmethod
    def from_options(cls, input_shape, n_filters, filter_length, n_conv_layers=1, n_fc_layers=1):
        args = cls._build_network(input_shape, n_conv_layers, n_fc_layers, n_filters, filter_length)

        args['name'] = cls.__class__.__name__

        return cls(**args)

    @classmethod
    def from_saved_model(cls, filepath):
        cls.__dict__ = load_model(filepath)
        cls.__class__ = DeepDNABind
        return cls

    def save(self, filepath, overwrite=True):
        self.__class__.__name__ = 'Model'
        super(DeepDNABind, self).save(filepath, overwrite=overwrite)

    @staticmethod
    def _build_network(input_shape, n_conv_layers, n_fc_layers, nb_filter, filter_length):
        assert len(input_shape) == 2, 'Unrecognizable input dimensions'
        assert K.image_dim_ordering() == 'tf', 'Theano dimension ordering not supported yet'

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
        max_pool = MaxPooling1D(pool_length=seq_length)(convs[-1])
        flat = Flatten()(max_pool)

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

        classifier = []
        for i in range(0, 6):
            classifier.append(Dense(4,
                                    init='glorot_uniform',
                                    activation='softmax',
                                    name='Classifier{}'.format(i + 1))(encoded))

        output = Merge(classifier, mode='concat')

        return {'input': inp, 'output': output}

    @staticmethod
    def _loss_function(y_true, y_pred):
        print(y_pred._keras_shape)
        print(y_true._keras_shape)
        loss = categorical_crossentropy(y_pred, y_true)
        return loss

    @staticmethod
    def mean_cat_acc(y_true, y_pred):
        cat_acc = categorical_accuracy(y_pred, y_true)
        return cat_acc

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
from keras.layers import GlobalMaxPooling1D
from keras.layers import Input
from keras.models import Model, load_model

from evolutron.networks.extra_layers import Convolution1D, MaxPooling1D, Dense  # To implement masking


class DeepDNABind(Model):
    def __init__(self, input, output, name=None):
        super(DeepDNABind, self).__init__(input, output, name)

        self.metrics = [self.mean_cat_acc]

    @classmethod
    def from_options(cls, input_shape, n_filters, filter_length, n_conv_layers=1, n_fc_layers=1):

        if not isinstance(n_filters, list):
            n_filters = [n_filters for _ in range(n_conv_layers)]
        if not isinstance(filter_length, list):
            filter_length = [filter_length for _ in range(n_conv_layers)]

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
        convs = []

        max_pools = [inp]
        for c in range(0, n_conv_layers - 1):
            convs.append(Convolution1D(nb_filter[c], filter_length[c],
                                       init='glorot_uniform',
                                       activation='sigmoid',
                                       border_mode='same',
                                       name='Conv{}'.format(c + 1))(max_pools[-1]))  # maybe add L1 regularizer
            max_pools.append(MaxPooling1D(pool_length=3)(convs[-1]))

        convs.append(Convolution1D(nb_filter[n_conv_layers - 1], filter_length[n_conv_layers - 1],
                                   init='glorot_uniform',
                                   activation='sigmoid',
                                   border_mode='same',
                                   name='Conv{}'.format(n_conv_layers))(max_pools[-1]))  # maybe add L1 regularizer

        max_pools.append(GlobalMaxPooling1D()(convs[-1]))

        #flat = Flatten()(max_pools[-1])

        # Fully-Connected encoding layers
        fc_enc = [max_pools[-1]]

        for d in range(0, n_fc_layers - 1):
            fc_enc.append(Dense(256,
                                init='glorot_uniform',
                                activation='sigmoid',
                                name='FCEnc{}'.format(d + 1))(fc_enc[-1]))

        encoded = Dense(64,
                        init='glorot_uniform',
                        activation='sigmoid',
                        name='FCEnc{}'.format(n_fc_layers))(fc_enc[-1])

        classifier = []
        for i in range(0, 6):
            classifier.append(Dense(4,
                                    init='glorot_uniform',
                                    activation='softmax',
                                    name='Classifier{}'.format(i + 1))(encoded))

        return {'input': inp, 'output': classifier}

    @property
    def _loss_function(self):
        return 'categorical_crossentropy'

    @property
    def mean_cat_acc(self):
        return 'categorical_accuracy'

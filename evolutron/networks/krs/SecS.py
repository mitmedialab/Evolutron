from keras.layers import Input, LSTM, Activation, Dropout, Masking
try:
    from .extra_layers import Convolution1D, MaxPooling1D, Dense, Flatten, Reshape
except Exception: #ImportError
    from extra_layers import Convolution1D, MaxPooling1D, Dense, Flatten, Reshape
from keras.models import Model, load_model
from keras.optimizers import SGD, Nadam
from keras.regularizers import l2, activity_l1
from keras.utils.visualize_util import model_to_dot
from keras.metrics import categorical_accuracy
from keras.objectives import categorical_crossentropy
from keras.callbacks import TensorBoard
import keras.backend as K

from IPython.display import SVG

import numpy as np
import argparse

class DeepCoDER(Model):
    def __init__(self, input, output, name=None):
        super(DeepCoDER, self).__init__(input, output, name)

    @classmethod
    def from_options(cls, aa_length, n_filters, filter_length, n_conv_layers=1, n_fc_layers=1, nb_categories=8):
        args = cls._build_network(aa_length, n_conv_layers, n_fc_layers, n_filters, filter_length, nb_categories)

        args['name'] = cls.__class__.__name__

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
    def _build_network(input_shape, n_conv_layers, n_fc_layers, nb_filter, filter_length,
                       nb_categories):
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
                                       name='Conv{}'.format(c + 1))(convs[-1]))

        # Max-pooling
        """
        if seq_length:
            max_pool = MaxPooling1D(pool_length=seq_length)(convs[-1])
            flat = Flatten()(max_pool)
        else:
            # max_pool = GlobalMaxPooling1D()(convs[-1])
            # flat = max_pool
            raise NotImplementedError('Sequence length must be known at this point. Pad and use mask.')
        """

        if n_conv_layers:
            lstm = LSTM(output_dim=seq_length, #input_shape=(-1, seq_length, alphabet),
                        return_sequences=False, W_regularizer=None)(convs[-1])
        else:
            lstm = LSTM(output_dim=seq_length,  # input_shape=(-1, seq_length, alphabet),
                        return_sequences=False, W_regularizer=None)(mask)

        #flat = Flatten()(lstm)

        dropout = Dropout(p=0.3)(lstm)

        # Fully-Connected layers
        fc = [Dense(seq_length*nb_categories,
                        init='glorot_uniform',
                        activation=None,
                        name='FCEnc1')(lstm)]

        for d in range(1, n_fc_layers):
            fc.append(Dense(seq_length*nb_categories,
                                init='glorot_uniform',
                                activation=None,
                                name='FCEnc{}'.format(d + 1))(fc[-1]))

        encoded = fc[-1]  # To access if model for encoding needed

        # Reshaping
        unflat = Reshape(target_shape=(seq_length, nb_categories))(encoded)

        # Softmaxing
        output = Activation(activation='softmax')(unflat)

        return {'input': inp, 'output': output}

    @staticmethod
    def _loss_function(y_true, y_pred):
        return K.mean(categorical_crossentropy(K.reshape(y_true, shape=(-1, 8)),
                                               K.reshape(y_pred, shape=(-1, 8))))

    @staticmethod
    def mean_cat_acc(y_true, y_pred):
        return categorical_accuracy(K.reshape(y_true, shape=(-1, 8)),
                                    K.reshape(y_pred, shape=(-1, 8)))

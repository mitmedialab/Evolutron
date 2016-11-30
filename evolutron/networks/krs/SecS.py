from keras.layers import Input, LSTM, Activation, Dropout, Masking, merge
try:
    from .extra_layers import Convolution1D, MaxPooling1D, Dense, Flatten, Reshape, Convolution2D, AtrousConvolution1D
except Exception: #ImportError
    from extra_layers import Convolution1D, MaxPooling1D, Dense, Flatten, Reshape, Convolution2D, AtrousConvolution1D
from keras.models import Model, load_model, model_from_json
from keras.optimizers import SGD, Nadam
from keras.regularizers import l2, activity_l1
from keras.utils.visualize_util import model_to_dot
from keras.metrics import categorical_accuracy
from keras.objectives import categorical_crossentropy, mse
from keras.callbacks import TensorBoard
import keras.backend as K

try:
    from evolutron.engine import DeepTrainer
    from evolutron.tools import load_dataset, Handle, shape
    from evolutron.networks import custom_layers
except ImportError:
    sys.path.insert(0, os.path.abspath('..'))
    from evolutron.engine import DeepTrainer
    from evolutron.tools import load_dataset, Handle, shape
    from evolutron.networks import custom_layers

import numpy as np
import argparse, sys, os, h5py

class DeepCoDER(Model):
    def __init__(self, input, output, name=None):
        super(DeepCoDER, self).__init__(input, output, name)

        self.metrics = [self.mean_cat_acc, ]


    @classmethod
    def from_options(cls, aa_length, n_filters, filter_length, n_conv_layers=1, n_fc_layers=1,
                     use_lstm=1, nb_categories=8, dilation=1):

        args = cls._build_network(aa_length, n_conv_layers, n_fc_layers, use_lstm, n_filters,
                                  filter_length, nb_categories)

        args['name'] = cls.__class__.__name__

        return cls(**args)

    @classmethod
    def from_saved_model(cls, filepath):
        # First load model architecture
        hf = h5py.File(filepath)
        model_config = hf.attrs['model_config'].decode('utf8')
        hf.close()
        model = model_from_json(model_config, custom_objects=custom_layers)

        #args['name'] = cls.__class__.__name__

        return cls(model.input, model.output, name='SecSDeepCoDER')

    def save(self, filepath, overwrite=True):
        self.__class__.__name__ = 'Model'
        super(DeepCoDER, self).save(filepath, overwrite=overwrite)

    @staticmethod
    def _build_network(input_shape, n_conv_layers, n_fc_layers, n_lstm, nb_filter,
                       filter_length, nb_categories, dilation=1):
        assert len(input_shape) == 2, 'Unrecognizable input dimensions'
        assert K.image_dim_ordering() == 'tf', 'Theano dimension ordering not supported yet'
        assert input_shape[1] in [20, 4, 22], 'Input dimensions error, check order'

        seq_length, alphabet = input_shape

        # Input LayerRO
        inp = Input(shape=input_shape, name='aa_seq')

        mask = Masking(mask_value=0.0)(inp)

        # Convolutional Layers
        convs = [AtrousConvolution1D(nb_filter, filter_length,
                                     atrous_rate=1,
                                     init='glorot_uniform',
                                     activation='relu',
                                     border_mode='same',
                                     name='Conv1')(mask)]

        for c in range(1, n_conv_layers):
            convs.append(AtrousConvolution1D(nb_filter, filter_length,
                                             atrous_rate=dilation^c,
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
            lstm1 = LSTM(output_dim=nb_categories,
                        return_sequences=True, W_regularizer=None)(convs[-1])
            lstm2 = LSTM(output_dim=nb_categories, go_backwards=True,
                        return_sequences=True, W_regularizer=None)(convs[-1])
            lstm3 = LSTM(output_dim=nb_categories, go_backwards=True,
                        return_sequences=True, W_regularizer=None)(lstm1)
        else:
            lstm1 = LSTM(output_dim=nb_categories,
                         return_sequences=True, W_regularizer=None)(mask)
            lstm2 = LSTM(output_dim=nb_categories, go_backwards=True,
                         return_sequences=True, W_regularizer=None)(mask)
            lstm3 = LSTM(output_dim=nb_categories, go_backwards=True,
                         return_sequences=True, W_regularizer=None)(lstm1)

        #merging forward and backward lstms
        merge_layer = merge([lstm1, lstm2], mode='sum')

        #flat = Reshape(target_shape=(-1, K.shape(lstm)[-1]))(lstm)
        if n_lstm == 1:
            flat = Flatten()(lstm1)
        elif n_lstm == 2:
            flat = Flatten()(lstm3)
        elif n_lstm == 11:
            flat = Flatten()(merge_layer)
        else:
            flat = Flatten()(convs[-1])

        #dropout = Dropout(p=0.3)(lstm)

        #maxpool = MaxPooling1D(pool_length=nb_categories)(flat)

        # Fully-Connected layers
        fc = [Dense(seq_length*nb_categories,
                    init='glorot_uniform',
                    activation=None,
                    name='FCEnc1')(flat)]

        for d in range(1, n_fc_layers):
            fc.append(Dense(seq_length*nb_categories,
                            init='glorot_uniform',
                            activation=None,
                            name='FCEnc{}'.format(d + 1))(fc[-1]))

        encoded = fc[-1]  # To access if model for encoding needed

        # Reshaping
        unflat = Reshape(target_shape=(seq_length, nb_categories))(encoded)

        #conv = Convolution1D(nb_categories, 10, border_mode='same')(lstm)

        # Softmaxing
        if n_fc_layers:
            output = Activation(activation='softmax')(unflat)
        elif n_lstm == 1:
            output = Activation(activation='softmax')(lstm1)
        elif n_lstm == 2:
            output = Activation(activation='softmax')(lstm3)
        elif n_lstm == 11:
            output = Activation(activation='softmax')(merge_layer)
        else:
            output = Activation(activation='softmax')(convs[-1])

        return {'input': inp, 'output': output}

    @staticmethod
    def _loss_function(y_true, y_pred):
        nb_categories = K.shape(y_true)[-1]
        return K.mean(categorical_crossentropy(K.reshape(y_true, shape=(-1, nb_categories)),
                                               K.reshape(y_pred, shape=(-1, nb_categories))))

    @staticmethod
    def mean_cat_acc(y_true, y_pred):
        nb_categories = K.shape(y_true)[-1]
        return categorical_accuracy(K.reshape(y_true, shape=(-1, nb_categories)),
                                    K.reshape(y_pred, shape=(-1, nb_categories)))

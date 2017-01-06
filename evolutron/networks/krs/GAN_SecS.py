from keras.layers import Input, LSTM, Activation, Dropout, Masking, merge, BatchNormalization, \
    ZeroPadding1D, TimeDistributed
from ..extra_layers import Convolution1D, MaxPooling1D, Dense, Flatten, Reshape, Convolution2D, \
        AtrousConvolution1D, Deconvolution1D, feedForwardLSTM, LocallyConnected1D
try:
    from .extra_metrics import mean_cat_acc, macro_precision, micro_precision, multiclass_precision, \
        multiclass_recall, macro_recall, micro_recall, multiclass_fmeasure, macro_fmeasure, micro_fmeasure
    from .extra_objectives import fmeasure_loss
except Exception: #ImportError
    from extra_metrics import mean_cat_acc, macro_precision, micro_precision, multiclass_precision, \
        multiclass_recall, macro_recall, micro_recall, multiclass_fmeasure, macro_fmeasure, micro_fmeasure
    from extra_objectives import fmeasure_loss
from keras.models import Model, load_model, model_from_json
from keras.regularizers import l2
from keras.metrics import categorical_accuracy, binary_accuracy, recall, precision, fmeasure
from keras.objectives import categorical_crossentropy, mse
import keras.backend as K

import numpy as np
import argparse, sys, os, h5py
from ast import literal_eval


try:
    from evolutron.engine import DeepTrainer
    from evolutron.tools import load_dataset, Handle, shape
    from evolutron.networks import custom_layers
except ImportError:
    sys.path.insert(0, os.path.abspath('..'))
    from evolutron.engine import DeepTrainer
    from evolutron.tools import load_dataset, Handle, shape
    from evolutron.networks import custom_layers


Alpha = 1


# Generator model for SecS GAN
class DeepSecS_Gen(Model):
    def __init__(self, input, output, name=None):
        super(DeepSecS_Gen, self).__init__(input, output, name)

        self.metrics = [self.mean_cat_acc]

    @classmethod
    def from_options(cls, aa_length, n_filters, filter_length, n_conv_layers=1, n_fc_layers=1,
                     use_lstm=1, nb_categories=8, dilation=1, nb_units=500, p=.4,
                     nb_lc_units=96, lc_filter_length=11, l=.1503, masking=True):

        args = cls._build_network(aa_length, n_conv_layers=n_conv_layers,
                                  n_fc_layers=n_fc_layers, n_lstm=use_lstm, nb_filters=n_filters,
                                  filter_length=filter_length, nb_categories=nb_categories,
                                  dilation=dilation, nb_units=nb_units, p=p,
                                  nb_lc_units=nb_lc_units, lc_filter_length=lc_filter_length, l=l,
                                  masking=masking)

        #args['name'] = cls.__class__.__name__
        args['name'] = 'Generator'

        return cls(**args)

    @classmethod
    def from_saved_model(cls, filepath):
        # First load model architecture
        hf = h5py.File(filepath)
        model_config = hf.attrs['model_config'].decode('utf8')
        hf.close()
        model = model_from_json(model_config, custom_objects=custom_layers)

        return cls(model.input, model.output, name='SecSDeepCoDER')

    def save(self, filepath, overwrite=True):
        self.__class__.__name__ = 'Model'
        super(DeepSecS_Gen, self).save(filepath, overwrite=overwrite)

    @staticmethod
    def _build_network(input_shape, n_conv_layers, n_fc_layers, n_lstm, nb_filters,
                       filter_length, nb_categories, dilation=1, nb_units=500, p=.4,
                       nb_lc_units=96, lc_filter_length=11, l=.1503, masking=True):
        assert len(input_shape) == 2, 'Unrecognizable input dimensions'
        assert K.image_dim_ordering() == 'tf', 'Theano dimension ordering not supported yet'

        seq_length, alphabet = input_shape

        try:
            nb_filters = literal_eval(nb_filters)
        except ValueError:
            nb_filters *= np.ones(n_conv_layers, dtype=np.int32)
        if type(nb_filters) == int:
            nb_filters *= np.ones(n_conv_layers, dtype=np.int32)

        filter_length = literal_eval(filter_length)
        nb_diff_lengths = len(filter_length) - 1

        drops = []

        # Input LayerRO
        inp = Input(shape=input_shape, name='G_aa_seq')

        if n_fc_layers or (not masking):
            mask = inp
        else:
            mask = Masking(mask_value=0.0, name='G_mask')(inp)

        # Convolutional Blocks
        if n_conv_layers > 0:
            # Multiscale convolutions
            multi_convs = [[]]
            for length in filter_length[1:]:
                multi_convs[0].append(AtrousConvolution1D(nb_filters[0], length,
                                                          atrous_rate=1,
                                                          init='glorot_uniform',
                                                          activation='linear',
                                                          border_mode='same',
                                                          W_regularizer=l2(l),
                                                          name='G_multiConv1_%d' % length)(mask))

            # Concat
            if nb_diff_lengths > 1:
                inner_concats = [merge(multi_convs[0], mode='concat', concat_axis=-1, name='G_inner_concat_1')]
            else:
                inner_concats = [multi_convs[0][-1]]

            # Regularizaion and non-linearity
            BNs = [BatchNormalization(name='G_BN_1')(inner_concats[-1])]
            Relus = [Activation('relu', name='G_relu_1')(BNs[-1])]

            # Uniscale convolution
            uni_convs = [AtrousConvolution1D(nb_filters[0], filter_length[0],
                                             atrous_rate=1,
                                             init='glorot_uniform',
                                             activation='linear',
                                             border_mode='same',
                                             W_regularizer=l2(l),
                                             name='G_uniConv1')(Relus[-1])]

            # Regularizaion and non-linearity
            BNs.append(BatchNormalization(name='G_BN_%d' % (len(BNs)+1))(uni_convs[-1]))
            Relus.append(Activation('relu', name='G_relu_%d' % (len(Relus)+1))(BNs[-1]))

            # Concat
            inner_concats.append(merge((Relus[-1], Relus[-2]), mode='concat', concat_axis=-1,
                                       name='G_inner_concat_%d' % (len(inner_concats)+1)))

            # Skip conv
            skip_convs = [Convolution1D(96, filter_length=1,
                                        init='glorot_uniform',
                                        activation='linear',
                                        border_mode='same',
                                        name='G_skipConv1')(mask)]

            # Regularizaion and non-linearity
            BNs.append(BatchNormalization(name='G_BN_%d' % (len(BNs)+1))(skip_convs[-1]))
            Relus.append(Activation('relu', name='G_relu_%d' % (len(Relus)+1))(BNs[-1]))

            # Final concat
            conv_blocks = [merge((Relus[-1], inner_concats[-1]), mode='concat', concat_axis=-1, name='conv_block_1')]

            # Dropout layer
            drops.append(Dropout(p, name='G_dropout_1')(conv_blocks[-1]))

        for c in range(1, n_conv_layers - 1):
            # Multiscale convolutions
            multi_convs.append([])
            for length in filter_length[1:]:
                multi_convs[c].append(AtrousConvolution1D(nb_filters[c], length,
                                                          atrous_rate=dilation ** (c-1),
                                                          init='glorot_uniform',
                                                          activation='linear',
                                                          border_mode='same',
                                                          W_regularizer=l2(l),
                                                          name='G_multiConv%d_%d' % (c+1, length))(drops[-1]))

            # Concat
            if nb_diff_lengths > 1:
                inner_concats.append(merge(multi_convs[c], mode='concat', concat_axis=-1,
                                           name='G_inner_concat_%d' % (len(inner_concats)+1)))
            else:
                inner_concats.append(multi_convs[c][-1])

            # Regularizaion and non-linearity
            BNs.append(BatchNormalization(name='G_BN_%d' % (len(BNs)+1))(inner_concats[-1]))
            Relus.append(Activation('relu', name='G_relu_%d' % (len(Relus)+1))(BNs[-1]))

            # Uniscale convolution
            uni_convs.append(AtrousConvolution1D(nb_filters[c], filter_length[0],
                                                 atrous_rate=1,
                                                 init='glorot_uniform',
                                                 activation='linear',
                                                 border_mode='same',
                                                 W_regularizer=l2(l),
                                                 name='G_uniConv%d' % (c+1))(Relus[-1]))

            # Regularizaion and non-linearity
            BNs.append(BatchNormalization(name='G_BN_%d' % (len(BNs)+1))(uni_convs[-1]))
            Relus.append(Activation('relu', name='G_relu_%d' % (len(Relus)+1))(BNs[-1]))

            # Concat
            inner_concats.append(merge((Relus[-1], Relus[-2]), mode='concat', concat_axis=-1,
                                       name='G_inner_concat_%d' % (len(inner_concats)+1)))

            # Skip conv
            skip_convs.append(Convolution1D(96, filter_length=1,
                                            init='glorot_uniform',
                                            activation='linear',
                                            border_mode='same',
                                            name='G_skipConv%d' % (c+1))(drops[-1]))

            # Regularizaion and non-linearity
            BNs.append(BatchNormalization(name='G_BN_%d' % (len(BNs)+1))(skip_convs[-1]))
            Relus.append(Activation('relu', name='G_relu_%d' % (len(Relus)+1))(BNs[-1]))

            # Final concat
            conv_blocks.append(merge((Relus[-1], inner_concats[-1]), mode='concat', concat_axis=-1,
                                     name='G_conv_block _%d' % (len(conv_blocks)+1)))

            # Dropout layer
            drops.append(Dropout(p, name='G_dropout_%d' % (len(drops)+1))(conv_blocks[-1]))

        if n_conv_layers:
            lstm_input = drops[-1]
        else:
            lstm_input = mask

        # First Bi-LSTM layer
        if n_lstm:
            forward_LSTMs = [feedForwardLSTM(output_dim=nb_units, return_sequences=True,
                                             go_backwards=False, W_regularizer=None,
                                             name='G_FW_LSTM_1')(lstm_input)]
            backward_LSTMs = [feedForwardLSTM(output_dim=nb_units, return_sequences=True,
                                              go_backwards=True, W_regularizer=None,
                                              name='G_BW_LSTM_1')(lstm_input)]
            concats = [merge((forward_LSTMs[-1], backward_LSTMs[-1]), mode='concat', concat_axis=-1,
                             name='G_concat_1')]
            drops.append(Dropout(p=p, name='G_dropout_%d' % (len(drops)+1))(concats[-1]))
            lstm_locally_connected = [TimeDistributed(Dense(output_dim=nb_units, activation='relu', W_regularizer=None,
                                                            input_shape=(seq_length, 2*nb_units),
                                                            name='G_dense_1'), name='G_lstm_LC_1')(drops[-1])]
            drops.append(Dropout(p=p, name='G_dropout_%d' % (len(drops)+1))(lstm_locally_connected[-1]))
            lstm_locally_connected.append(TimeDistributed(Dense(output_dim=nb_units, activation='relu',
                                                                W_regularizer=None,
                                                                input_shape=(seq_length, 2*nb_units),
                                                                name='G_dense_%d' % (len(lstm_locally_connected)+1)),
                                                          name='G_lstm_LC_%d' % (len(lstm_locally_connected)+1))
                                          (drops[-1]))
            drops.append(Dropout(p=p, name='G_dropout_%d' % (len(drops)+1))(lstm_locally_connected[-1]))

        # Add more Bi-LSTM layers
        for i in range(1, n_lstm):
            forward_LSTMs.append(feedForwardLSTM(output_dim=nb_units, return_sequences=True,
                                                 go_backwards=False, W_regularizer=None,
                                                 name='G_FW_LSTM_%d' % (len(forward_LSTMs)+1))
                                 (lstm_locally_connected[-1]))
            backward_LSTMs.append(feedForwardLSTM(output_dim=nb_units, return_sequences=True,
                                                  go_backwards=True, W_regularizer=None,
                                                  name='G_BW_LSTM_%d' % (len(backward_LSTMs)+1))
                                  (lstm_locally_connected[-1]))
            concats.append(merge((forward_LSTMs[-1], backward_LSTMs[-1]), mode='concat', concat_axis=-1,
                                 name='G_concat_%d' % (len(concats)+1)))
            drops.append(Dropout(p=p, name='G_dropout_%d' % (len(drops)+1))(concats[-1]))
            lstm_locally_connected.append(TimeDistributed(Dense(output_dim=nb_units, activation='relu',
                                                                W_regularizer=None,
                                                                input_shape=(seq_length, 2*nb_units),
                                                                name='G_dense_%d' % (len(lstm_locally_connected)+1)),
                                                          name='G_lstm_LC_%d' % (len(lstm_locally_connected)+1))
                                          (drops[-1]))
            drops.append(Dropout(p=p, name='G_dropout_%d' % (len(drops)+1))(lstm_locally_connected[-1]))
            lstm_locally_connected.append(TimeDistributed(Dense(output_dim=nb_units, activation='relu',
                                                                W_regularizer=None,
                                                                input_shape=(seq_length, 2*nb_units),
                                                                name='G_dense_%d' % (len(lstm_locally_connected)+1)),
                                                          name='G_lstm_LC_%d' % (len(lstm_locally_connected)+1))
                                          (drops[-1]))
            drops.append(Dropout(p=p, name='G_dropout_%d' % (len(drops)+1))(lstm_locally_connected[-1]))

        # Locally connected layers
        if n_fc_layers:
            #padded = [ZeroPadding1D((lc_filter_length-1)//2)(drops[-1])]
            locally_connected = [Convolution1D(nb_filter=nb_lc_units, filter_length=lc_filter_length,
                                               activation='relu', W_regularizer=l2(l),
                                               border_mode='same', name='G_LC_1')(drops[-1])]
            drops.append(Dropout(p=p, name='G_dropout_%d' % (len(drops)+1))(locally_connected[-1]))

        for i in range(1, n_fc_layers):
            #padded.append(ZeroPadding1D((lc_filter_length - 1) // 2)(drops[-1]))
            locally_connected.append(Convolution1D(nb_filter=nb_lc_units, filter_length=lc_filter_length,
                                                   activation='relu', W_regularizer=l2(l),
                                                   border_mode='same', name='G_LC_%d' % (len(locally_connected)))
                                     (drops[-1]))
            drops.append(Dropout(p=p, name='G_dropout_%d' % (len(drops)+1))(locally_connected[-1]))

        # Fixing output dims
        if n_fc_layers:
            final_lc = TimeDistributed(Dense(output_dim=nb_categories, init='glorot_uniform',
                                             activation='linear', name='G_final_LC'))(locally_connected[-1])
        else:
            final_lc = TimeDistributed(Dense(output_dim=nb_categories, init='glorot_uniform',
                                             activation='linear', name='G_final_LC'))(drops[-1])

        # Softmaxing
        output = Activation(activation='softmax', name='G_output')(final_lc)

        return {'input': inp, 'output': output}

    @staticmethod
    def _loss_function(y_true, y_pred):
        nb_categories = K.shape(y_true)[-1]
        return K.mean(categorical_crossentropy(K.reshape(y_true, shape=(-1, nb_categories)),
                                               K.reshape(y_pred, shape=(-1, nb_categories))))

    @staticmethod
    def mean_cat_acc(y_true, y_pred):
        return mean_cat_acc(y_true, y_pred)

    @staticmethod
    def multiclass_precision(y_true, y_pred):
        return multiclass_precision(y_true, y_pred)

    @staticmethod
    def macro_precision(y_true, y_pred):
        return macro_precision(y_true, y_pred)

    @staticmethod
    def micro_precision(y_true, y_pred):
        return micro_precision(y_true, y_pred)

    @staticmethod
    def multiclass_recall(y_true, y_pred):
        return multiclass_recall(y_true, y_pred)

    @staticmethod
    def macro_recall(y_true, y_pred):
        return macro_recall(y_true, y_pred)

    @staticmethod
    def micro_recall(y_true, y_pred):
        return micro_recall(y_true, y_pred)

    @staticmethod
    def multiclass_fmeasure(y_true, y_pred):
        return multiclass_fmeasure(y_true, y_pred)

    @staticmethod
    def macro_fmeasure(y_true, y_pred):
        return macro_fmeasure(y_true, y_pred)

    @staticmethod
    def micro_fmeasure(y_true, y_pred):
        return micro_fmeasure(y_true, y_pred)


# Discriminator model for SecS GAN
class DeepSecS_Dis(Model):
    def __init__(self, input, output, name=None):
        super(DeepSecS_Dis, self).__init__(input, output, name)

        self.metrics = [binary_accuracy]

    @classmethod
    def from_options(cls, aa_length, n_filters, filter_length, n_conv_layers=1, n_fc_layers=1,
                     use_lstm=1, nb_categories=8, dilation=1, nb_units=500, p=.4,
                       nb_lc_units=96, lc_filter_length=11, l=.1503):

        args = cls._build_network(aa_length, n_conv_layers=n_conv_layers,
                                  n_fc_layers=n_fc_layers, n_lstm=use_lstm, nb_filters=n_filters,
                                  filter_length=filter_length, nb_categories=nb_categories,
                                  dilation=dilation, nb_units=nb_units, p=p,
                                  nb_lc_units=nb_lc_units, lc_filter_length=lc_filter_length, l=l)

        #args['name'] = cls.__class__.__name__ + '_D'
        args['name'] = 'Discriminator'

        return cls(**args)

    @classmethod
    def from_saved_model(cls, filepath):
        # First load model architecture
        hf = h5py.File(filepath)
        model_config = hf.attrs['model_config'].decode('utf8')
        hf.close()
        model = model_from_json(model_config, custom_objects=custom_layers)

        return cls(model.input, model.output, name='SecSDeepCoDER')

    def save(self, filepath, overwrite=True):
        self.__class__.__name__ = 'Model'
        super(DeepSecS_Dis, self).save(filepath, overwrite=overwrite)

    @staticmethod
    def _build_network(input_shape, n_conv_layers, n_fc_layers, n_lstm, nb_filters,
                       filter_length, nb_categories, dilation=1, nb_units=500, p=.4,
                       nb_lc_units=96, lc_filter_length=11, l=.1503):
        assert len(input_shape) == 2, 'Unrecognizable input dimensions'
        assert K.image_dim_ordering() == 'tf', 'Theano dimension ordering not supported yet'

        seq_length, alphabet = input_shape

        try:
            nb_filters = literal_eval(nb_filters)
        except ValueError:
            nb_filters *= np.ones(n_conv_layers, dtype=np.int32)
        if type(nb_filters) == int:
            nb_filters *= np.ones(n_conv_layers, dtype=np.int32)

        filter_length = literal_eval(filter_length)
        nb_diff_lengths = len(filter_length) - 1

        drops = []

        # Input LayerRO
        inp = Input(shape=input_shape, name='D_aa_seq')

        """if n_fc_layers:
            mask = inp
        else:
            mask = Masking(mask_value=0.0)(inp)"""

        # Convolutional Blocks
        if n_conv_layers > 0:
            # Multiscale convolutions
            multi_convs = [[]]
            for length in filter_length[1:]:
                multi_convs[0].append(AtrousConvolution1D(nb_filters[0], length,
                                                          atrous_rate=1,
                                                          init='glorot_uniform',
                                                          activation='linear',
                                                          border_mode='same',
                                                          W_regularizer=l2(l),
                                                          name='D_multiConv1_%d' % length)(inp))

            # Concat
            if nb_diff_lengths > 1:
                inner_concats = [merge(multi_convs[0], mode='concat', concat_axis=-1)]
            else:
                inner_concats = [multi_convs[0][-1]]

            # Regularizaion and non-linearity
            BNs = [BatchNormalization()(inner_concats[-1])]
            Relus = [Activation('relu')(BNs[-1])]

            # Uniscale convolution
            uni_convs = [AtrousConvolution1D(nb_filters[0], filter_length[0],
                                             atrous_rate=1,
                                             init='glorot_uniform',
                                             activation='linear',
                                             border_mode='same',
                                             W_regularizer=l2(l),
                                             name='D_uniConv1')(Relus[-1])]

            # Regularizaion and non-linearity
            BNs.append(BatchNormalization()(uni_convs[-1]))
            Relus.append(Activation('relu')(BNs[-1]))

            # Concat
            inner_concats.append(merge((Relus[-1], Relus[-2]), mode='concat', concat_axis=-1))

            # Skip conv
            skip_convs = [Convolution1D(96, filter_length=1,
                                        init='glorot_uniform',
                                        activation='linear',
                                        border_mode='same',
                                        name='D_skipConv1')(inp)]

            # Regularizaion and non-linearity
            BNs.append(BatchNormalization()(skip_convs[-1]))
            Relus.append(Activation('relu')(BNs[-1]))

            # Final concat
            conv_blocks = [merge((Relus[-1], inner_concats[-1]), mode='concat', concat_axis=-1)]

            # Dropout layer
            drops.append(Dropout(p)(conv_blocks[-1]))

        for c in range(1, n_conv_layers - 1):
            # Multiscale convolutions
            multi_convs.append([])
            for length in filter_length[1:]:
                multi_convs[c].append(AtrousConvolution1D(nb_filters[c], length,
                                                          atrous_rate=dilation ** (c-1),
                                                          init='glorot_uniform',
                                                          activation='linear',
                                                          border_mode='same',
                                                          W_regularizer=l2(l),
                                                          name='D_multiConv%d_%d' % (c+1, length))(drops[-1]))

            # Concat
            if nb_diff_lengths > 1:
                inner_concats.append(merge(multi_convs[c], mode='concat', concat_axis=-1))
            else:
                inner_concats.append(multi_convs[c][-1])

            # Regularizaion and non-linearity
            BNs.append(BatchNormalization()(inner_concats[-1]))
            Relus.append(Activation('relu')(BNs[-1]))

            # Uniscale convolution
            uni_convs.append(AtrousConvolution1D(nb_filters[c], filter_length[0],
                                                 atrous_rate=1,
                                                 init='glorot_uniform',
                                                 activation='linear',
                                                 border_mode='same',
                                                 W_regularizer=l2(l),
                                                 name='D_uniConv%d' % (c+1))(Relus[-1]))

            # Regularizaion and non-linearity
            BNs.append(BatchNormalization()(uni_convs[-1]))
            Relus.append(Activation('relu')(BNs[-1]))

            # Concat
            inner_concats.append(merge((Relus[-1], Relus[-2]), mode='concat', concat_axis=-1))

            # Skip conv
            skip_convs.append(Convolution1D(96, filter_length=1,
                                            init='glorot_uniform',
                                            activation='linear',
                                            border_mode='same',
                                            name='D_skipConv%d' % (c+1))(drops[-1]))

            # Regularizaion and non-linearity
            BNs.append(BatchNormalization()(skip_convs[-1]))
            Relus.append(Activation('relu')(BNs[-1]))

            # Final concat
            conv_blocks.append(merge((Relus[-1], inner_concats[-1]), mode='concat', concat_axis=-1))

            # Dropout layer
            drops.append(Dropout(p)(conv_blocks[-1]))

        if n_conv_layers:
            lstm_input = drops[-1]
        else:
            lstm_input = inp

        # First Bi-LSTM layer
        if n_lstm:
            forward_LSTMs = [feedForwardLSTM(output_dim=nb_units, return_sequences=True,
                                             go_backwards=False, W_regularizer=None)(lstm_input)]
            backward_LSTMs = [feedForwardLSTM(output_dim=nb_units, return_sequences=True,
                                              go_backwards=True, W_regularizer=None)(lstm_input)]
            concats = [merge((forward_LSTMs[-1], backward_LSTMs[-1]), mode='concat', concat_axis=-1)]
            drops.append(Dropout(p=p)(concats[-1]))
            lstm_locally_connected = [TimeDistributed(Dense(output_dim=nb_units, activation='relu', W_regularizer=None,
                                                            input_shape=(seq_length, 2*nb_units)))(drops[-1])]
            drops.append(Dropout(p=p)(lstm_locally_connected[-1]))
            lstm_locally_connected.append(TimeDistributed(Dense(output_dim=nb_units, activation='relu',
                                                                W_regularizer=None,
                                                                input_shape=(seq_length, 2*nb_units)))(drops[-1]))
            drops.append(Dropout(p=p)(lstm_locally_connected[-1]))

        # Add more Bi-LSTM layers
        for i in range(1, n_lstm):
            forward_LSTMs.append(feedForwardLSTM(output_dim=nb_units, return_sequences=True,
                                                 go_backwards=False, W_regularizer=None)(lstm_locally_connected[-1]))
            backward_LSTMs.append(feedForwardLSTM(output_dim=nb_units, return_sequences=True,
                                                  go_backwards=True, W_regularizer=None)(lstm_locally_connected[-1]))
            concats.append(merge((forward_LSTMs[-1], backward_LSTMs[-1]), mode='concat', concat_axis=-1))
            drops.append(Dropout(p=p)(concats[-1]))
            lstm_locally_connected.append(TimeDistributed(Dense(output_dim=nb_units, activation='relu',
                                                                W_regularizer=None,
                                                                input_shape=(seq_length, 2*nb_units)))(drops[-1]))
            drops.append(Dropout(p=p)(lstm_locally_connected[-1]))
            lstm_locally_connected.append(TimeDistributed(Dense(output_dim=nb_units, activation='relu',
                                                                W_regularizer=None,
                                                                input_shape=(seq_length, 2*nb_units)))(drops[-1]))
            drops.append(Dropout(p=p)(lstm_locally_connected[-1]))

        # Locally connected layers
        if n_fc_layers:
            #padded = [ZeroPadding1D((lc_filter_length-1)//2)(drops[-1])]
            locally_connected = [Convolution1D(nb_filter=nb_lc_units, filter_length=lc_filter_length,
                                               activation='relu', W_regularizer=l2(l),
                                               border_mode='same')(drops[-1])]
            drops.append(Dropout(p=p)(locally_connected[-1]))

        for i in range(1, n_fc_layers):
            #padded.append(ZeroPadding1D((lc_filter_length - 1) // 2)(drops[-1]))
            locally_connected.append(Convolution1D(nb_filter=nb_lc_units, filter_length=lc_filter_length,
                                                   activation='relu', W_regularizer=l2(l),
                                                   border_mode='same')(drops[-1]))
            drops.append(Dropout(p=p)(locally_connected[-1]))

        # Computing the probability for each AA
        if n_fc_layers:
            final_lc = TimeDistributed(Dense(output_dim=1, init='glorot_uniform',
                                                  activation='sigmoid'))(locally_connected[-1])
        else:
            final_lc = TimeDistributed(Dense(output_dim=1, init='glorot_uniform',
                                             activation='sigmoid'))(drops[-1])

        # Computing the final output
        flat = Flatten()(final_lc)
        output = Dense(output_dim=1, init='glorot_uniform', activation='sigmoid')(flat)

        return {'input': inp, 'output': output}

    @staticmethod
    def _loss_function(y_true, y_pred):
        return K.binary_crossentropy(y_pred, y_true)


# Combined model for SecS GAN
class DeepSecS_Gen_Dis(Model):
    def __init__(self, input, output, name=None):
        super(DeepSecS_Gen_Dis, self).__init__(input, output, name)

        self.metrics = None

    @classmethod
    def from_options(cls, aa_length, n_filters, filter_length, n_conv_layers=1, n_fc_layers=1,
                     use_lstm=1, nb_categories=8, dilation=1, nb_units=500, p=.4,
                     nb_lc_units=96, lc_filter_length=11, l=.1503):
        generator = DeepSecS_Gen.from_options(aa_length=aa_length, n_filters=n_filters, filter_length=filter_length,
                                              n_conv_layers=n_conv_layers, n_fc_layers=n_fc_layers, use_lstm=use_lstm,
                                              nb_categories=nb_categories, dilation=dilation, nb_units=nb_units, p=p,
                                              nb_lc_units=nb_lc_units, lc_filter_length=lc_filter_length, l=l,
                                              masking=False)

        discriminator = DeepSecS_Dis.from_options(aa_length=(aa_length[0], aa_length[1]+8), n_filters=n_filters,
                                                  filter_length=filter_length, n_conv_layers=n_conv_layers,
                                                  n_fc_layers=n_fc_layers, use_lstm=use_lstm,
                                                  nb_categories=nb_categories, dilation=dilation,
                                                  nb_units=nb_units, p=p, nb_lc_units=nb_lc_units,
                                                  lc_filter_length=lc_filter_length, l=l)

        args = cls._build_network(aa_length, generator, discriminator)

        #args['name'] = cls.__class__.__name__ + '_GD'
        args['name'] = 'Gen_Dis'

        return generator, discriminator, cls(**args)

    @classmethod
    def from_saved_model(cls, filepath):
        # First load model architecture
        hf = h5py.File(filepath)
        model_config = hf.attrs['model_config'].decode('utf8')
        hf.close()
        model = model_from_json(model_config, custom_objects=custom_layers)

        return cls(model.input, model.output, name='SecSDeepCoDER')

    def save(self, filepath, overwrite=True):
        self.__class__.__name__ = 'Model'
        super(DeepSecS_Gen_Dis, self).save(filepath, overwrite=overwrite)

    @staticmethod
    def _build_network(input_shape, generator, discriminator):
        assert len(input_shape) == 2, 'Unrecognizable input dimensions'
        assert K.image_dim_ordering() == 'tf', 'Theano dimension ordering not supported yet'

        seq_length, alphabet = input_shape

        # Input LayerRO
        inp = Input(shape=input_shape, name='aa_seq')

        gen = generator(inp)

        concat = merge((inp, gen), mode='concat', concat_axis=-1)

        dis = discriminator(concat)

        return {'input': inp, 'output': [gen, dis]}

    @staticmethod
    def _loss_function(y_true, y_pred):
        SecS_pred = y_pred[0, :]
        SecS_true = y_true[0, :]

        dis_pred = y_pred[1, :]

        nb_categories = K.shape(SecS_true)[-1]
        cat_cross_ent_loss = K.mean(categorical_crossentropy(K.reshape(SecS_true, shape=(-1, nb_categories)),
                                                             K.reshape(SecS_pred, shape=(-1, nb_categories))))

        discriminator_loss = -K.mean(K.log(1-dis_pred+K.epsilon()))

        return discriminator_loss + Alpha*cat_cross_ent_loss

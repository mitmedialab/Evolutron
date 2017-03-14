import keras.backend as K
from keras.layers import Input, LSTM, Activation, Masking
from keras.metrics import categorical_accuracy
from keras.models import Model, load_model, model_from_json
from keras.objectives import categorical_crossentropy, mse
from keras.regularizers import l2

from networks.krs.extra_layers import Convolution1D, MaxPooling1D, Dense, Flatten, Deconvolution1D

try:
    from metrics.krs.extra_metrics import mean_cat_acc
except Exception: #ImportError
    from extra_metrics import mean_cat_acc

import sys, os, h5py

try:
    from evolutron.engine import DeepTrainer
    from evolutron.tools import load_dataset, Handle, shape
    from evolutron.networks import custom_layers
except ImportError:
    sys.path.insert(0, os.path.abspath('..'))
    from evolutron.engine import DeepTrainer
    from evolutron.tools import load_dataset, Handle, shape
    from evolutron.networks import custom_layers


class DeepEmbed(Model):
    def __init__(self, input, output, name=None):
        super().__init__(input, output, name)

        self.metrics = [self.mean_cat_acc, ]


    @classmethod
    def from_options(cls, aa_length, n_filters, filter_length, n_conv_layers=1, n_fc_layers=0,
                     nb_categories=8, dilation=1, embedder='conv'):

        if embedder == 'conv':
            args = cls._build_conv_network(input_shape=aa_length, n_conv_layers=n_conv_layers,
                                           n_fc_layers=n_fc_layers, nb_filters=n_filters,
                                           dilation=dilation, filter_length=filter_length,
                                           nb_categories=nb_categories)
        elif embedder == 'rnn':
            args = cls._build_rnn_network(input_shape=aa_length, nb_categories=nb_categories)
        else:
            args = cls._build_brnn_network(input_shape=aa_length, nb_categories=nb_categories, nb_filters=n_filters)

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
        super(DeepEmbed, self).save(filepath, overwrite=overwrite)

    @staticmethod
    def _build_conv_network(input_shape, n_conv_layers, n_fc_layers, nb_filters,
                            filter_length, nb_categories, dilation=1):
        assert len(input_shape) == 2, 'Unrecognizable input dimensions'
        assert K.image_dim_ordering() == 'tf', 'Theano dimension ordering not supported yet'
        assert input_shape[1] in [20, 4, 22, 44], 'Input dimensions error, check order'

        seq_length, alphabet = input_shape

        # Input LayerRO
        inp = Input(shape=input_shape, name='aa_seq')

        mask = Masking(mask_value=0.0)(inp)

        # Convolutional Layers
        convs = [Convolution1D(nb_filter=alphabet,
                               filter_length=filter_length,
                                init='glorot_uniform',
                                activation='relu',
                                border_mode='same',
                                name='Conv1')(mask)]

        for c in range(1, n_conv_layers-1):
            convs.append(Convolution1D(nb_filters, filter_length,
                                         init='glorot_uniform',
                                         activation='relu',
                                         border_mode='same',
                                         name='Conv{}'.format(c + 1))(convs[-1]))

        # Embedding layer for output
        encoded = (Convolution1D(nb_filter=nb_categories,
                                 filter_length=filter_length,
                                 init='glorot_uniform',
                                 activation='relu',
                                 border_mode='same',
                                 name='encoded')(convs[-1]))
        #encoded = convs[-1]

        # De-convolutions
        deconvs = [Deconvolution1D(bound_conv_layer=encoded._keras_history[0],
                                   activation='relu',
                                   border_mode='same',
                                   name='Deconv1')(encoded)]

        for c in range(n_conv_layers-2, 0, -1):
            deconvs.append(Deconvolution1D(bound_conv_layer=convs[c]._keras_history[0],
                                           activation='relu',
                                           border_mode='same',
                                           name='Deconv{}'.format(n_conv_layers - c))(deconvs[-1]))

        deconvs.append(Deconvolution1D(bound_conv_layer=convs[0]._keras_history[0],
                                       activation='relu',
                                       border_mode='same',
                                       name='Deconv{}'.format(n_conv_layers))(deconvs[-1]))

        # Softmaxing
        output = Activation(activation='softmax')(deconvs[-1])

        return {'input': inp, 'output': output}

    @staticmethod
    def _build_rnn_network(input_shape, nb_categories):
        assert len(input_shape) == 2, 'Unrecognizable input dimensions'
        assert K.image_dim_ordering() == 'tf', 'Theano dimension ordering not supported yet'
        assert input_shape[1] in [20, 4, 22, 44], 'Input dimensions error, check order'

        seq_length, alphabet = input_shape

        # Input LayerRO
        inp = Input(shape=input_shape, name='aa_seq')

        mask = Masking(mask_value=0.0)(inp)



        # Embedding layer for output
        encoded = LSTM(output_dim=nb_categories,
                       return_sequences=True,
                       W_regularizer=None,
                       name='encoded')(mask)
        #encoded = lstms[-1]

        # De-lstms
        delstms = [LSTM(output_dim=alphabet,
                        return_sequences=True,
                        W_regularizer=None,
                        go_backwards=True)(encoded)]

        # Softmaxing
        output = Activation(activation='softmax')(delstms[-1])

        return {'input': inp, 'output': output}

    @staticmethod
    def _build_brnn_network(input_shape, nb_categories, nb_filters):
        assert len(input_shape) == 2, 'Unrecognizable input dimensions'
        assert K.image_dim_ordering() == 'tf', 'Theano dimension ordering not supported yet'
        assert input_shape[1] in [20, 4, 22, 44], 'Input dimensions error, check order'

        seq_length, alphabet = input_shape

        # Input LayerRO
        inp = Input(shape=input_shape, name='aa_seq')

        mask = Masking(mask_value=0.0)(inp)

        lstms = [LSTM(output_dim=nb_filters,
                      return_sequences=True,
                      W_regularizer=None)(mask)]



        # Embedding layer for output
        encoded = (LSTM(output_dim=nb_categories,
                        return_sequences=True,
                        W_regularizer=None,
                        go_backwards=True,
                        name='encoded')(lstms[-1]))

        #encoded = lstms[-1]

        # De-lstms
        delstms = [LSTM(output_dim=nb_filters,
                        return_sequences=True,
                        W_regularizer=None,
                        go_backwards=True)(encoded)]

        delstms.append(LSTM(output_dim=alphabet,
                            return_sequences=True,
                            W_regularizer=None,
                            go_backwards=False)(delstms[-1]))

        # Softmaxing
        output = Activation(activation='softmax')(delstms[-1])

        return {'input': inp, 'output': output}

    @staticmethod
    def _loss_function(y_true, y_pred):
        """nb_categories = K.shape(y_true)[-1]
        return K.mean(categorical_crossentropy(K.reshape(y_true, shape=(-1, nb_categories)),
                                               K.reshape(y_pred, shape=(-1, nb_categories))))"""
        return mse(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def mean_cat_acc(y_true, y_pred):
        return mean_cat_acc(y_true, y_pred)


class DeepFAM(Model):
    def __init__(self, input, output, name=None):
        super().__init__(input, output, name)

        self.metrics = [self.mean_cat_acc, ]


    @classmethod
    def from_options(cls, aa_length, n_filters, filter_length, n_conv_layers=1, n_fc_layers=0,
                     nb_families=None, dilation=1, embedder='fam'):

        args = cls._build_network(input_shape=aa_length, n_conv_layers=n_conv_layers,
                                  n_fc_layers=n_fc_layers, nb_filters=n_filters,
                                  dilation=dilation, filter_length=filter_length,
                                  nb_families=nb_families)

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
        super(DeepFAM, self).save(filepath, overwrite=overwrite)

    @staticmethod
    def _build_network(input_shape, n_conv_layers, n_fc_layers, nb_filters,
                       filter_length, nb_families, dilation=1):
        assert len(input_shape) == 2, 'Unrecognizable input dimensions'
        assert K.image_dim_ordering() == 'tf', 'Theano dimension ordering not supported yet'
        assert input_shape[1] in [20, 4, 22, 44], 'Input dimensions error, check order'

        seq_length, alphabet = input_shape
        print(input_shape)

        # Input LayerRO
        inp = Input(shape=input_shape, name='aa_seq')

        mask = Masking(mask_value=0.0)(inp)

        # Convolutional Layers
        convs = [Convolution1D(nb_filter=nb_filters,
                               filter_length=filter_length,
                                init='glorot_uniform',
                                activation='relu',
                                border_mode='same',
                                name='Conv1')(mask)]

        for c in range(1, n_conv_layers-1):
            convs.append(Convolution1D(nb_filters, filter_length,
                                         init='glorot_uniform',
                                         activation='relu',
                                         border_mode='same',
                                         name='Conv{}'.format(c + 1))(convs[-1]))



        # Embedding layer for output
        encoded = Convolution1D(nb_filter=nb_filters,
                                filter_length=filter_length,
                                init='glorot_uniform',
                                activation='relu',
                                border_mode='same',
                                name='encoded')(convs[-1])

        # Max-pooling and flattening
        if seq_length:
            max_pool = MaxPooling1D(pool_length=seq_length)(encoded)
            flat = Flatten()(max_pool)
        else:
            raise NotImplementedError('Sequence length must be known at this point. Pad and use mask.')

        # Softmaxing
        output = Dense(nb_families,
                       activation='softmax')(flat)

        return {'input': inp, 'output': output}

    @staticmethod
    def _loss_function(y_true, y_pred):
        return categorical_crossentropy(y_true, y_pred)

    @staticmethod
    def mean_cat_acc(y_true, y_pred):
        return categorical_accuracy(y_true, y_pred)


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

        for c in range(1, n_conv_layers-1):
            convs.append(Convolution1D(nb_filter, filter_length,
                                       init='glorot_uniform',
                                       activation='relu',
                                       border_mode='same',
                                       name='Conv{}'.format(c + 1))(convs[-1]))  # maybe add L1 regularizer

        if n_conv_layers > 1:
            convs.append(Convolution1D(2, filter_length,
                                       init='glorot_uniform',
                                       activation='relu',
                                       border_mode='same',
                                       name='Conv{}'.format(n_conv_layers))(convs[-1]))

        deconvs = [Deconvolution1D(convs[-1]._keras_history[0],
                                           activation='relu',
                                           name='Deconv{}'.format(n_conv_layers))(convs[-1])]

        # Deconvolution
        for c in range(n_conv_layers - 2, 0, -1):
            deconvs.append(Deconvolution1D(convs[c]._keras_history[0],
                                           activation='relu',
                                           name='Deconv{}'.format(c + 1))(deconvs[-1]))

        decoded = Deconvolution1D(convs[0]._keras_history[0],
                                  apply_mask=True,
                                  activation='sigmoid',
                                  name='Deconv1')(deconvs[-1])

        return {'input': inp, 'output': decoded}

    @staticmethod
    def _loss_function(inp, decoded):
        # y_true = K.reshape(inp, (-1, K.shape(inp)[-1]))
        # y_pred = K.reshape(decoded, (-1, K.shape(inp)[-1]))
        # loss = K.mean(categorical_crossentropy(y_true, y_pred))
        loss = mse(y_true=inp, y_pred=decoded)
        return loss

    @staticmethod
    def mean_cat_acc(inp, decoded):
        y_true = K.reshape(inp, (-1, K.shape(inp)[-1]))
        y_pred = K.reshape(decoded, (-1, K.shape(inp)[-1]))
        cat_acc = categorical_accuracy(y_true, y_pred)
        return cat_acc


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

        for c in range(1, n_conv_layers-1):
            convs.append(Convolution1D(nb_filter, filter_length,
                                       init='glorot_uniform',
                                       activation='relu',
                                       border_mode='same',
                                       name='Conv{}'.format(c + 1))(convs[-1]))

        if n_conv_layers > 1:
            convs.append(Convolution1D(2, filter_length,
                                       init='glorot_uniform',
                                       activation='relu',
                                       border_mode='same',
                                       name='Conv{}'.format(n_conv_layers))(convs[-1]))

        flat = Flatten()(convs[-1])

        classifier = Dense(output_dim,
                           init='glorot_uniform',
                           activation='softmax',
                           W_regularizer=l2(2),
                           name='Classifier')(flat)

        return {'input': inp, 'output': classifier}

    @staticmethod
    def _loss_function(y_true, y_pred):
        loss = categorical_crossentropy(y_true=y_true, y_pred=y_pred)
        return loss

    @staticmethod
    def mean_cat_acc(y_true, y_pred):
        cat_acc = categorical_accuracy(y_pred, y_true)
        return cat_acc


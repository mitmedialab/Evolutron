from keras.layers import Input, LSTM, Activation, Dropout, Masking
try:
    from .extra_layers import Convolution1D, MaxPooling1D, Dense, Flatten, Reshape
except Exception: #ImportError
    from extra_layers import Convolution1D, MaxPooling1D, Dense, Flatten, Reshape
from keras.models import Model
from keras.optimizers import SGD, Nadam
from keras.regularizers import l2, activity_l1
from keras.utils.visualize_util import model_to_dot
from keras.metrics import categorical_accuracy
import keras.backend as K

from IPython.display import SVG

import numpy as np
import argparse

class DeepCoDER(Model):
    def __init__(self, input, output, name=None):
        super(DeepCoDER, self).__init__(input, output, name)

    @classmethod
    def from_options(cls, aa_length, n_filters, filter_length, n_conv_layers=1, n_fc_layers=1):
        args = cls._build_network(aa_length, n_conv_layers, n_fc_layers, n_filters, filter_length)

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

        #Not implemented yet
        """
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
        """

        lstm = LSTM(output_dim=nb_categories, input_shape=(-1, seq_length, alphabet),
                    return_sequences=True, W_regularizer=None)(mask)

        flat = Flatten()(lstm)

        dropout = Dropout(p=0.3)(flat)

        # Fully-Connected layers
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

        # Reshaping and unpooling
        if seq_length:
            unflat = Reshape(max_pool._keras_shape[1:])(fc_dec[-1])
        else:
            unflat = Reshape((1, fc_dec[-1]._keras_shape[-1]))(fc_dec[-1])

        deconvs = [Unpooling1D(max_pool._keras_history[0], name='Unpooling')(unflat)]

        return {'input': inp, 'output': decoded}

    @staticmethod
    def _loss_function(inp, decoded):
        # y_true = K.reshape(inp, (-1, K.shape(inp)[-1]))
        # y_pred = K.reshape(decoded, (-1, K.shape(inp)[-1]))
        # loss = K.mean(categorical_crossentropy(y_true, y_pred))
        loss = mean_squared_error(y_true=inp, y_pred=decoded)
        return loss

    @staticmethod
    def mean_cat_acc(inp, decoded):
        y_true = K.reshape(inp, (-1, K.shape(inp)[-1]))
        y_pred = K.reshape(decoded, (-1, K.shape(inp)[-1]))
        cat_acc = categorical_accuracy(y_true, y_pred)
        return cat_acc


def SecStructure(args):
    dataset, secondary_structure = load_dataset(data_id, padded=True)

    num_examples = dataset.shape[0]
    example_len = dataset.shape[1]
    num_channels = dataset.shape[2]

    print(dataset.shape)

    input_aa_seq = Input(shape=(example_len, num_channels))

    masking = Masking(mask_value=0)(input_aa_seq)

    lstm = LSTM(output_dim=args.num_categories, input_shape=(None, example_len, num_channels),
                return_sequences=True, W_regularizer=l2(0.1))(masking)

    flat = Flatten()(lstm)

    dropout = Dropout(p=0.3)(flat)

    dense = Dense(output_dim=example_len*args.num_categories,
                  activity_regularizer=activity_l1(0.1))(dropout)

    reshape = Reshape(target_shape=(example_len, args.num_categories))(dense)

    output = Activation(activation='softmax')(reshape)

    model = Model(input=input_aa_seq, output=output)

    sgd = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

    nadam = Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    model.compile(optimizer=nadam, loss=vector_categorical_crossentropy, metrics=['categorical_accuracy'])

    model.summary()

    model.fit(dataset, secondary_structure,
              batch_size=args.batch_size,
              nb_epoch=args.num_epochs,
              shuffle=True,
              validation_split=.2)

    model.save('tmp')

    #SVG(model_to_dot(model).create(prog='dot', format='svg'))

    print(dataset[0:2, :, :])
    print(secondary_structure[0:2, :, :])
    print(model.predict(dataset[0:2, :, :]))


# custom metrics: categorical accuracy for a vector of predictions
"""def vector_categorical_accuracy(y_true, y_pred):
    #ToDo: implement in Theano
    return K.mean(K.equal([K.argmax(y_true[i, :, :], axis=-1) for i in K.int_shape(y_true)[0]],
                          [K.argmax(y_pred[i, :, :], axis=-1) for i in K.int_shape(y_true)[0]]))"""


def vector_categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(K.permute_dimensions(
        K.batch_flatten(K.permute_dimensions((2,0,1))), (1,0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process html patents')
    parser.add_argument('--size', '-s', type=str, default='full',
                        help='dataset size')
    parser.add_argument('--num_categories', '-c', type=int, default=8,
                        help='how many categories (3/8)?')
    parser.add_argument('--num_epochs', '-e', type=int, default=10,
                        help='how many epochs?')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01,
                        help='learning rate?')
    parser.add_argument('--batch_size', '-b', type=int, default=128,
                        help='batch size?')

    args = parser.parse_args()

    if args.size == 'full':
        data_id = 'SecS'
    elif args.size == 'small':
        data_id = 'smallSecS'
    else:
        print('Unknown data size: should be full or small')

    SecStructure(args)

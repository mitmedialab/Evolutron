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
from keras.models import Model
from keras.layers import Input, Convolution1D, MaxPooling1D, Dense, Flatten, Reshape, UpSampling1D
from keras.optimizers import SGD


class CoDER:
    def __init__(self, aa_length, batch_size, filters, filter_size):
        self.inp = {'aa_seq': Input(shape=(aa_length, 20))}  # Assuming tf dimension ordering

        # self.targets = {'aa_seq': ten.tensor3('aa_rec_seq', dtype=theano.config.floatX)}

        self.aa_length = aa_length
        self.batch_size = batch_size
        self.filters = filters
        self.filter_size = filter_size

        self.network = self.build_network()

        self.layers = self.network.layers

        self.handle = 'krCoDER'

    def build_network(self):
        conv1 = Convolution1D(self.filters, self.filter_size,
                              init='glorot_uniform',  # change that for gaussian
                              activation='relu',
                              border_mode='same')(self.inp['aa_seq'])  # maybe add L1 regularizer

        max_pool = MaxPooling1D(pool_length=self.aa_length)(conv1)

        flat = Flatten()(max_pool)

        dense = Dense(max(flat._keras_shape[1] / 2, 100), init='glorot_uniform', activation='linear')(flat)

        encoded = Dense(100, init='glorot_uniform', activation='sigmoid')(dense)

        dedense = Dense(max(flat._keras_shape[1] / 2, 100), init='glorot_uniform', activation='linear')(encoded)

        dedense = Dense(flat._keras_shape[1], init='glorot_uniform', activation='linear')(dedense)

        unflat = Reshape(max_pool._keras_shape[1:])(dedense)

        unpool = UpSampling1D(length=self.aa_length)(unflat)

        decoded = Convolution1D(20, 50, activation='sigmoid', border_mode='same')(unpool)

        autoencoder = Model(input=self.inp['aa_seq'], output=decoded)

        return autoencoder

    @property
    def optimizer(self, deterministic=False):
        return SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    @property
    def loss(self):
        return 'mse'

#
# class DeepCoDER:
#     def __init__(self, aa_length, batch_size, filters, filter_size):
#         self.inp = {'aa_seq': ten.tensor3('aa_seq', dtype=theano.config.floatX)}
#         self.targets = {'aa_seq': ten.tensor3('aa_rec_seq', dtype=theano.config.floatX)}
#
#         self.aa_length = aa_length
#         self.batch_size = batch_size
#         self.filters = filters
#         self.filter_size = filter_size
#
#         self.layers = self.build_network()
#
#         self.network = self.layers['output']
#
#         self.handle = 'CoDER_test'
#
#     def build_network(self):
#         network = {'input': lasagne.layers.InputLayer(input_var=self.inp['aa_seq'],
#                                                       shape=(self.batch_size, 20, self.aa_length),
#                                                       name='Input')}
#
#         # Convolutional layer with M motifs of size m.
#         network['conv1'] = lasagne.layers.Conv1DLayer(network['input'],
#                                                       num_filters=self.filters,
#                                                       filter_size=self.filter_size,
#                                                       flip_filters=False,
#                                                       nonlinearity=None,
#                                                       W=lasagne.init.GlorotUniform('relu'),
#                                                       stride=self.filter_size - 2,
#                                                       pad='valid',
#                                                       name='Conv1')
#
#         network['conv2'] = lasagne.layers.Conv1DLayer(network['conv1'],
#                                                       num_filters=self.filters,
#                                                       filter_size=self.filter_size,
#                                                       flip_filters=False,
#                                                       nonlinearity=None,
#                                                       W=lasagne.init.GlorotUniform('relu'),
#                                                       stride=self.filter_size - 2,
#                                                       pad='valid',
#                                                       name='Conv2')
#
#         network['conv3'] = lasagne.layers.Conv1DLayer(network['conv2'],
#                                                       num_filters=self.filters,
#                                                       filter_size=self.filter_size,
#                                                       flip_filters=False,
#                                                       nonlinearity=None,
#                                                       W=lasagne.init.GlorotUniform('relu'),
#                                                       stride=self.filter_size - 2,
#                                                       pad='valid',
#                                                       name='Conv3')
#
#         network['conv_non_lin'] = lasagne.layers.NonlinearityLayer(network['conv3'],
#                                                                    nonlinearity=lasagne.nonlinearities.rectify,
#                                                                    name='nonlin')
#
#         # Max-pooling layer to select best motif score for each motif.
#
#         # network['maxpool'] = lasagne.layers.MaxPool1DLayer(network['conv'],
#         #                                                    pool_size=100,
#         #                                                    stride=1,
#         #                                                    name='MaxPool')
#         network['maxpool'] = lasagne.layers.GlobalPoolLayer(network['conv_non_lin'],
#                                                             pool_function=ten.max,
#                                                             name='MaxPool')
#
#         network['FC1'] = lasagne.layers.DenseLayer(network['maxpool'],
#                                                    num_units=self.filters,
#                                                    nonlinearity=lasagne.nonlinearities.sigmoid,
#                                                    name='FC1')
#
#         network['inv_FC1'] = lasagne.layers.InverseLayer(network['FC1'], network['FC1'], name='inv_hidden')
#
#         network['inv_pool'] = lasagne.layers.InverseLayer(network['inv_FC1'], network['maxpool'], name='inv_pool')
#
#         network['inv_conv3'] = lasagne.layers.InverseLayer(network['inv_pool'], network['conv3'], name='inv_conv3')
#
#         network['inv_conv2'] = lasagne.layers.InverseLayer(network['inv_conv3'], network['conv2'], name='inv_conv2')
#
#         network['inv_conv1'] = lasagne.layers.InverseLayer(network['inv_conv2'], network['conv1'], name='inv_conv1')
#
#         network['output'] = network['inv_conv1']
#
#         return network
#
#     def build_loss(self, deterministic=False):
#         prediction = lasagne.layers.get_output(self.network, deterministic=deterministic)
#
#         code_dist = prediction.dimshuffle((0, 2, 1)).reshape((prediction.shape[0] * prediction.shape[2], 20))
#
#         true_dist = ten.argmax(
#             self.targets['aa_seq'].dimshuffle((0, 2, 1))
#                 .reshape((self.targets['aa_seq'].shape[0] * self.targets['aa_seq'].shape[2], 20)), axis=1)
#
#         # loss = lasagne.objectives.categorical_crossentropy(code_dist, true_dist)
#         loss = lasagne.objectives.squared_error(prediction, self.targets['aa_seq'])
#         # lasagne.regularization.regularize_layer_params(self.layers['conv'], lasagne.regularization.l2)
#
#         loss = loss.mean()
#
#         # acc = lasagne.objectives.categorical_accuracy(code_dist, true_dist)
#
#         # acc = acc.mean()
#         acc = loss.mean()
#         return loss, acc, prediction

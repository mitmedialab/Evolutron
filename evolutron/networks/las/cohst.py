# coding=utf-8
"""
    Implementation of a Convolutional protein Motif Extraction Tool (CoMET) in theano.
"""
import lasagne
import theano
import theano.tensor as ten


# noinspection PyDictCreation
class CoHST:
    def __init__(self, pad_size, filters, filter_size):
        self.inp = {'aa_seq': ten.tensor3('aa_seq', dtype=theano.config.floatX)}
        self.targets = {'class': ten.matrix('targets', dtype=theano.config.floatX)}

        self.pad_size = pad_size
        self.filters = filters
        self.filter_size = filter_size

        self.layers = self.build_network()
        self.network = self.layers['output']

        self.handle = 'CoHST'

    def build_network(self):
        network = {}

        network['input'] = lasagne.layers.InputLayer(input_var=self.inp['aa_seq'],
                                                     shape=(None, 20, self.pad_size),
                                                     name='Input')

        # Convolutional layer with M motifs of size m.
        network['conv'] = lasagne.layers.Conv1DLayer(network['input'],
                                                     num_filters=self.filters,
                                                     filter_size=self.filter_size,
                                                     flip_filters=False,
                                                     nonlinearity=lasagne.nonlinearities.rectify,
                                                     W=lasagne.init.GlorotUniform('relu'),
                                                     pad='valid',
                                                     stride=1,
                                                     name='Conv1')

        # Max-pooling layer to select best motif score for each motif.

        network['maxpool'] = lasagne.layers.GlobalPoolLayer(network['conv'],
                                                            pool_function=ten.max,
                                                            name='MaxPool')

        network['FC1'] = lasagne.layers.DenseLayer(network['maxpool'],
                                                   num_units=self.filters,
                                                   nonlinearity=lasagne.nonlinearities.sigmoid,
                                                   W=lasagne.init.GlorotUniform('relu'),
                                                   name='FC1')

        network['dropout1'] = lasagne.layers.DropoutLayer(network['FC1'],
                                                          p=.1,
                                                          name='Drop1')

        network['FC2'] = lasagne.layers.DenseLayer(network['dropout1'],
                                                   num_units=self.filters,
                                                   nonlinearity=lasagne.nonlinearities.sigmoid,
                                                   W=lasagne.init.GlorotUniform(),
                                                   name='FC2')
        #
        network['dropout2'] = lasagne.layers.DropoutLayer(network['FC2'],
                                                          p=.1,
                                                          name='Drop2')

        network['FC3'] = lasagne.layers.DenseLayer(network['dropout2'],
                                                   num_units=self.filters,
                                                   nonlinearity=lasagne.nonlinearities.sigmoid,
                                                   W=lasagne.init.GlorotUniform(),
                                                   name='FC3')
        #
        #     #
        #         network['dropout3'] = lasagne.layers.DropoutLayer(network['FC3'],
        #                                                           p=.1,
        #                                                           name='Drop3')
        #
        #         network['FC4'] = lasagne.layers.DenseLayer(network['dropout3'],
        #                                                    num_units=self.filters,
        #                                                    nonlinearity=lasagne.nonlinearities.sigmoid,
        #                                                    W=lasagne.init.GlorotUniform(),
        #                                                    name='FC4')
        #
        #
        # #
        #         network['dropout4'] = lasagne.layers.DropoutLayer(network['FC4'],
        #                                                   p=.1,
        #                                                   name='Drop4')

        network['output'] = lasagne.layers.DenseLayer(network['FC3'],
                                                      num_units=1,
                                                      nonlinearity=lasagne.nonlinearities.sigmoid,
                                                      W=lasagne.init.GlorotUniform(),
                                                      name='Output')

        return network

    def build_loss(self, deterministic=False):
        prediction = lasagne.layers.get_output(self.network, deterministic=deterministic)

        loss = lasagne.objectives.binary_crossentropy(prediction, ten.cast(self.targets['class'], dtype='int32'))

        loss = loss.mean()

        acc = lasagne.objectives.binary_accuracy(prediction, ten.cast(self.targets['class'], dtype='int32'))

        acc = acc.mean()

        return loss, acc, prediction

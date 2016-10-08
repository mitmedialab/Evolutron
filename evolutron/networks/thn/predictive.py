# coding=utf-8
import lasagne
import theano
import theano.tensor as ten


# noinspection PyDictCreation
class ConvType2p:
    def __init__(self, pad_size, filters, filter_size):
        self.inp = ten.tensor3('data', dtype=theano.config.floatX)
        self.targets = ten.matrix('targets', dtype=theano.config.floatX)

        self.pad_size = pad_size
        self.filters = filters
        self.filter_size = filter_size

        self.layers = self.build_network()
        self.network = self.layers['output']

    def build_network(self):
        network = {}

        network['input'] = lasagne.layers.InputLayer(input_var=self.inp,
                                                     shape=(None, 20, self.pad_size),
                                                     name='Input')

        # Convolutional layer with M motifs of size m.
        network['conv1'] = lasagne.layers.Conv1DLayer(network['input'],
                                                      num_filters=self.filters,
                                                      filter_size=self.filter_size,
                                                      flip_filters=False,
                                                      nonlinearity=lasagne.nonlinearities.rectify,
                                                      W=lasagne.init.GlorotUniform('relu'),
                                                      stride=3,
                                                      name='Conv1')

        network['conv2'] = lasagne.layers.Conv1DLayer(network['conv1'],
                                                      num_filters=self.filters,
                                                      filter_size=self.filter_size,
                                                      flip_filters=False,
                                                      nonlinearity=lasagne.nonlinearities.rectify,
                                                      W=lasagne.init.GlorotUniform('relu'),
                                                      stride=3,
                                                      name='Conv2')

        network['conv3'] = lasagne.layers.Conv1DLayer(network['conv2'],
                                                      num_filters=self.filters,
                                                      filter_size=self.filter_size,
                                                      flip_filters=False,
                                                      nonlinearity=lasagne.nonlinearities.rectify,
                                                      W=lasagne.init.GlorotUniform('relu'),
                                                      stride=3,
                                                      name='Conv3')

        network['maxpool'] = lasagne.layers.GlobalPoolLayer(network['conv3'],
                                                            pool_function=ten.max,
                                                            name='MaxPool')

        network['FC_1'] = lasagne.layers.DenseLayer(network['maxpool'],
                                                    num_units=self.filters,
                                                    nonlinearity=lasagne.nonlinearities.sigmoid,
                                                    name='FC1')

        network['base1'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network['FC_1'], p=.1, name='Drop.1'),
                                                     num_units=4,
                                                     nonlinearity=lasagne.nonlinearities.softmax,
                                                     name='FC_Base1')

        network['base2'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network['FC_1'], p=.1, name='Drop.1'),
                                                     num_units=4,
                                                     nonlinearity=lasagne.nonlinearities.softmax,
                                                     name='FC_Base2')

        network['base3'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network['FC_1'], p=.1, name='Drop.1'),
                                                     num_units=4,
                                                     nonlinearity=lasagne.nonlinearities.softmax,
                                                     name='FC_Base3')

        network['base4'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network['FC_1'], p=.1, name='Drop.1'),
                                                     num_units=4,
                                                     nonlinearity=lasagne.nonlinearities.softmax,
                                                     name='FC_Base4')

        network['base5'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network['FC_1'], p=.1, name='Drop.1'),
                                                     num_units=4,
                                                     nonlinearity=lasagne.nonlinearities.softmax,
                                                     name='FC_Base5')

        network['base6'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network['FC_1'], p=.1, name='Drop.1'),
                                                     num_units=4,
                                                     nonlinearity=lasagne.nonlinearities.softmax,
                                                     name='FC_Base6')

        network['output'] = lasagne.layers.concat([network['base' + str(i)] for i in range(1, 7)], name='Output')

        return network

    def build_loss(self, deterministic=False):
        prediction = lasagne.layers.get_output(self.network, deterministic=deterministic)

        loss = lasagne.objectives.categorical_crossentropy(prediction.reshape((-1, 6, 4)),
                                                           self.targets.reshape((-1, 6, 4)))
        loss = loss.mean()

        acc = lasagne.objectives.categorical_accuracy(prediction.reshape((-1, 6, 4)), self.targets.reshape((-1, 6, 4)))
        acc = acc.mean()

        return loss, acc, prediction


# noinspection PyDictCreation
class ConvZFb1h:
    def __init__(self, pad_size, filters, filter_size):
        self.inp = ten.tensor3('data', dtype=theano.config.floatX)
        self.targets = ten.matrix('targets', dtype=theano.config.floatX)

        self.pad_size = pad_size
        self.filters = filters
        self.filter_size = filter_size

        self.layers = self.build_network()
        self.network = self.layers['output']

    def build_network(self):
        network = {}

        network['input'] = lasagne.layers.InputLayer(input_var=self.inp,
                                                     shape=(None, 20, self.pad_size),
                                                     name='Input')

        # Convolutional layer with M motifs of size m.
        network['conv'] = lasagne.layers.Conv1DLayer(network['input'],
                                                     num_filters=self.filters,
                                                     filter_size=self.filter_size,
                                                     flip_filters=False,
                                                     nonlinearity=lasagne.nonlinearities.rectify,
                                                     W=lasagne.init.GlorotUniform('relu'),
                                                     stride=1,
                                                     name='Conv1')

        # network['conv2'] = lasagne.layers.Conv1DLayer(network['conv'],
        #                                              num_filters=self.filters,
        #                                              filter_size=self.filter_size,
        #                                              flip_filters=False,
        #                                              nonlinearity=lasagne.nonlinearities.rectify,
        #                                              W=lasagne.init.GlorotUniform('relu'),
        #                                              stride=1,
        #                                              name='Conv2')
        #
        # network['conv3'] = lasagne.layers.Conv1DLayer(network['conv2'],
        #                                               num_filters=self.filters,
        #                                               filter_size=self.filter_size,
        #                                               flip_filters=False,
        #                                               nonlinearity=lasagne.nonlinearities.rectify,
        #                                               W=lasagne.init.GlorotUniform('relu'),
        #                                               stride=1,
        #                                               name='Conv3')

        # Max-pooling layer to select best motif score for each motif.

        network['maxpool'] = lasagne.layers.GlobalPoolLayer(network['conv'],
                                                            pool_function=ten.max,
                                                            name='MaxPool')

        # network['FC1'] = lasagne.layers.DenseLayer(network['maxpool'],
        #                                            num_units=self.filters,
        #                                            name='FC1')

        network['base1'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network['maxpool'], p=.1, name='Drop.1'),
                                                     num_units=4,
                                                     nonlinearity=lasagne.nonlinearities.softmax,
                                                     name='FC_Base1')

        network['base2'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network['maxpool'], p=.1, name='Drop.1'),
                                                     num_units=4,
                                                     nonlinearity=lasagne.nonlinearities.softmax,
                                                     name='FC_Base2')

        network['base3'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network['maxpool'], p=.1, name='Drop.1'),
                                                     num_units=4,
                                                     nonlinearity=lasagne.nonlinearities.softmax,
                                                     name='FC_Base3')

        network['base4'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network['maxpool'], p=.1, name='Drop.1'),
                                                     num_units=4,
                                                     nonlinearity=lasagne.nonlinearities.softmax,
                                                     name='FC_Base4')

        network['output'] = lasagne.layers.concat([network['base' + str(i)] for i in xrange(1, 5)], name='Output')

        return network

    def build_loss(self, deterministic=False):
        prediction = lasagne.layers.get_output(self.network, deterministic=deterministic)

        # loss = lasagne.objectives.squared_error(prediction, self.targets)

        loss = lasagne.objectives.categorical_crossentropy(prediction.reshape((-1, 4, 4)),
                                                           self.targets.reshape((-1, 4, 4)))
        loss = loss.mean()

        acc = lasagne.objectives.categorical_accuracy(prediction.reshape((-1, 4, 4)), self.targets.reshape((-1, 4, 4)))
        acc = acc.mean()

        return loss, acc, prediction


# noinspection PyDictCreation
class ConvM6a:
    def __init__(self, pad_size, filters, filter_size, binary):
        self.inp = ten.tensor3('data', dtype=theano.config.floatX)
        self.targets = ten.vector('targets', dtype=theano.config.floatX)

        self.pad_size = pad_size
        self.filters = filters
        self.filter_size = filter_size
        self.binary = binary

        self.layers = self.build_network()
        self.network = self.layers['output']

    def build_network(self):
        network = {}

        network['input'] = lasagne.layers.InputLayer(input_var=self.inp,
                                                     shape=(None, 20, self.pad_size),
                                                     name='Input')

        # Convolutional layer with M motifs of size m.
        network['conv'] = lasagne.layers.Conv1DLayer(network['input'],
                                                     num_filters=self.filters,
                                                     filter_size=self.filter_size,
                                                     flip_filters=False,
                                                     nonlinearity=lasagne.nonlinearities.rectify,
                                                     W=lasagne.init.GlorotUniform('relu'),
                                                     stride=1,
                                                     name='Conv1')

        # Max-pooling layer to select best motif score for each motif.

        network['maxpool'] = lasagne.layers.GlobalPoolLayer(network['conv'],
                                                            pool_function=ten.max,
                                                            name='MaxPool')

        network['FC1'] = lasagne.layers.DenseLayer(network['maxpool'],
                                                   num_units=self.filters,
                                                   nonlinearity=lasagne.nonlinearities.sigmoid,
                                                   name='FC1')
        # Drop-out
        # network['dropout'] = lasagne.layers.dropout(network['maxpool'], p=.1, name='Drop.1')

        if self.binary:
            network['output'] = lasagne.layers.DenseLayer(network['FC1'],
                                                          num_units=1,
                                                          nonlinearity=lasagne.nonlinearities.sigmoid,
                                                          name='Output')
        else:
            # Fully connected softmax layer
            network['output'] = lasagne.layers.DenseLayer(network['FC1'],
                                                          num_units=3,
                                                          nonlinearity=lasagne.nonlinearities.softmax,
                                                          name='Output')

        return network

    def build_loss(self, deterministic=False):
        prediction = lasagne.layers.get_output(self.network, deterministic=deterministic)

        if self.binary:
            # loss = lasagne.objectives.binary_crossentropy(prediction, ten.cast(self.targets, 'int32'))
            loss = lasagne.objectives.squared_error(prediction, self.targets)
            loss = loss.mean()

            acc = lasagne.objectives.binary_accuracy(prediction, ten.cast(self.targets, 'int32'))
            acc = acc.mean()
        else:
            loss = lasagne.objectives.categorical_crossentropy(prediction, ten.cast(self.targets, 'int32'))
            loss = loss.mean()

            acc = lasagne.objectives.categorical_accuracy(prediction, ten.cast(self.targets, 'int32'))
            acc = acc.mean()

        return loss, acc, prediction

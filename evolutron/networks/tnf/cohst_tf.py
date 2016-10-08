# coding=utf-8
"""
    Implementation of a Convolutional protein Motif Extraction Tool (CoMET) in theano.
"""
from collections import OrderedDict

import tensorflow as tf
from evolutron.tools.net_tools import *

"""
    Implementation of a Convolutional protein Motif Extraction Tool (CoMET) in Tensorflow.
"""


class tfCoHST(object):
    def __init__(self, pad_size, filters, filter_size, num_conv_layers=1, num_fc_layers=2, keep_prob=0.9):
        self.inp = tf.placeholder(tf.float32, shape=[None, 20, pad_size], name='data')
        self.targets = tf.placeholder(tf.float32, name='targets')

        self.pad_size = pad_size
        self.num_conv_layers = num_conv_layers
        self.num_fc_layers = num_fc_layers
        self.keep_prob = keep_prob

        if len(filters) == num_conv_layers + num_fc_layers:
            self.filters = filters
        elif len(filters) == 1:
            self.filters = [filters[0] for _ in range(num_conv_layers + num_fc_layers)]
        else:
            print('filters len doesn\'t fit number of layers')
            exit()

        if len(filter_size) == num_conv_layers:
            self.filter_size = filter_size
        elif len(filter_size) == 1:
            self.filter_size = [filter_size for _ in range(num_conv_layers)]
        else:
            print('filter_size len doesn\'t fit number of layers')
            exit()

        self.layers = self.build_network()
        self.network = self.layers['output']

    def build_network(self):
        network = OrderedDict({'input': self.inp})

        for i in range(self.num_conv_layers):
            if i == 0:
                network['conv' + str(i + 1)] = conv1d_layer(network['input'],
                                                            num_filters=self.filters[i],
                                                            filter_size=self.filter_size[i],
                                                            name='conv' + str(i + 1),
                                                            W_relu=True)
                # no non-linearity layer!!!!!
                tf.histogram_summary('conv' + str(i + 1) + '/activations', network['conv' + str(i + 1)])
            else:
                network['conv' + str(i + 1)] = conv1d_layer(network['conv' + str(i)],
                                                            num_filters=self.filters[i],
                                                            filter_size=self.filter_size[i],
                                                            name='conv' + str(i + 1),
                                                            W_relu=True)
                tf.histogram_summary('conv' + str(i + 1) + '/activations',
                                     network['conv' + str(i + 1)])  # keep_dims=True)

        network['maxpool' + str(i + 1)] = max_pool(network['conv' + str(i + 1)], name='maxpool' + str(i + 1))
        tf.histogram_summary('maxpool' + str(i + 1) + '/activations', network['maxpool' + str(i + 1)])

        keep_prob = tf.placeholder_with_default(self.keep_prob, [], name='keep_prob')

        for i in range(self.num_fc_layers):
            if i == 0:
                network['FC' + str(i + 1)] = FC_layer(network['maxpool' + str(self.num_conv_layers)],
                                                      num_units=self.filters[self.num_conv_layers + i],
                                                      nonlinearity=tf.nn.sigmoid,
                                                      name='FC' + str(i + 1),
                                                      W_relu=1)
                tf.histogram_summary('FC' + str(i + 1) + '/activations', network['FC' + str(i + 1)])

                network['dropout' + str(i + 1)] = tf.nn.dropout(network['FC' + str(i + 1)],
                                                                keep_prob=keep_prob,
                                                                name='Drop' + str(i + 1))
                tf.histogram_summary('dropout' + str(i + 1) + '/activations', network['dropout' + str(i + 1)])
            elif i == self.num_fc_layers - 1:
                network['FC' + str(i + 1)] = FC_layer(network['dropout' + str(i)],
                                                      num_units=self.filters[self.num_conv_layers + i],
                                                      nonlinearity=tf.nn.sigmoid,
                                                      W=tf.contrib.layers.xavier_initializer(),
                                                      name='FC' + str(i + 1))
                tf.histogram_summary('FC' + str(i + 1) + '/activations', network['FC' + str(i + 1)])
            else:
                network['FC' + str(i + 1)] = FC_layer(network['dropout' + str(i)],
                                                      num_units=self.filters[self.num_conv_layers + i],
                                                      nonlinearity=tf.nn.sigmoid,
                                                      name='FC' + str(i + 1),
                                                      W_relu=1)
                tf.histogram_summary('FC' + str(i + 1) + '/activations', network['FC' + str(i + 1)])

                network['dropout' + str(i + 1)] = tf.nn.dropout(network['FC' + str(i + 1)],
                                                                keep_prob=keep_prob,
                                                                name='Drop' + str(i + 1))
                tf.histogram_summary('dropout' + str(i + 1) + '/activations', network['dropout' + str(i + 1)])

        network['output'] = FC_layer(network['FC' + str(self.num_fc_layers)],
                                     num_units=1,
                                     nonlinearity=tf.nn.sigmoid,
                                     W=tf.contrib.layers.xavier_initializer(),
                                     name='output')
        tf.histogram_summary('output/activations', network['output'])

        return network

    def build_loss(self):
        prediction = self.network
        targets = tf.cast(self.targets, dtype='float32')

        loss = tf.reduce_mean(bin_crossentropy(prediction, targets), name='loss')
        tf.scalar_summary('loss', loss)

        acc = tf.reduce_mean(bin_accuracy(prediction, targets), name='acc')
        tf.scalar_summary('accuracy', acc)

        return loss, acc, prediction

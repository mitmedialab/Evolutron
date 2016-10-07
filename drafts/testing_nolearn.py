#!/usr/bin/env python

from __future__ import print_function

import numpy as np
from lasagne.objectives import squared_error
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from evolutron.tools import load_dataset
from evolutron.tools.io_tools import type2p
from evolutron.trainers.thn import build_network_type2p as bdd

print("Loading data...")
raw_data = type2p(padded=True)
x_train, y_train = load_dataset(raw_data, shuffled=True)

# size = x_train.shape[2]

# # Store values in shared variables to be transferred to GPU memory
# train_set_x = theano.shared(np.asarray(x_train, dtype=theano.config.floatX), borrow=True)
# train_set_y = theano.shared(np.asarray(y_train, dtype=theano.config.floatX), borrow=True)
#
# # compute number of minibatches for training and validation
# n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

# # Prepare Theano variables for inputs and targets as well as index to minibatch
# inputs = ten.tensor3('inputs')
# targets = ten.fmatrix('targets')
# index = ten.lscalar()  # index to a [mini]batch

# network= build_network()
network = bdd()

net = NeuralNet(layers=network, max_epochs=10,
                update=nesterov_momentum,
                update_learning_rate=0.01,
                update_momentum=0.975,
                objective_loss_function=squared_error,
                verbose=1,
                regression=True)

net.fit(x_train, np.asarray(y_train))

# # Create a loss expression for training, i.e., a scalar objective we want to minimize
# prediction = lasagne.layers.get_output(network)
# loss = lasagne.objectives.squared_error(prediction, targets)
# loss = loss.mean()
#
# # Maybe add weight decay here, see lasagne.regularization.
#
# # Create update expressions for training
# # Here, I use Stochastic Gradient Descent (SGD) with Nesterov momentum
# params = lasagne.layers.get_all_params(network, trainable=True)
# updates = lasagne.updates.nesterov_momentum(
#     loss, params, learning_rate=0.01, momentum=0.9)
#
# # Compile a function performing a training step on a mini-batch
# train_fn = theano.function([index], loss,
#                            updates=updates,
#                            givens={
#                                inputs: train_set_x[index * batch_size: (index + 1) * batch_size],
#                                targets: train_set_y[index * batch_size: (index + 1) * batch_size]
#                            })
#
# # Finally, launch the training loop.
# print("Starting training Final Model")
# epoch = 0
#
# # We iterate over epochs:
# while epoch < num_epochs:
#     epoch += 1
#     # In each epoch, we do a full pass over the training data:
#     train_err = 0
#     for minibatch_index in xrange(n_train_batches):
#         train_err += train_fn(minibatch_index)
#
#     print("Epoch {0}, training loss:\t\t{1:.6f}".format(epoch, train_err / n_train_batches))
#
# print('trained final model')
# hyperparameters = np.asarray([num_epochs, batch_size, filters, filter_size])
# np.savez_compressed('models/type2p_pad/{0}_{1}_{2}_{3}.npz'.format(num_epochs, batch_size, filters, filter_size),
#                     hyperparameters,
#                     *lasagne.layers.get_all_param_values(network))
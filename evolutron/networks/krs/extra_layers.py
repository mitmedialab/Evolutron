# -*- coding: utf-8 -*-
import functools
import keras.backend as K
import numpy as np
from keras.layers import regularizers, activations, constraints, initializers
from keras.activations import relu
from keras.engine import InputSpec
from keras.initializers import glorot_uniform
from keras.layers.recurrent import Recurrent

import keras.layers as native
from keras.layers.convolutional import _Conv


class Convolution1D(native.Conv1D):
    def __init__(self, nb_filter, filter_length, **kwargs):
        self.supports_masking = True
        super(Convolution1D, self).__init__(nb_filter, filter_length, **kwargs)


class LocallyConnected1D(native.LocallyConnected1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(LocallyConnected1D, self).__init__(**kwargs)


class Convolution2D(native.Conv2D):
    def __init__(self, nb_filter, filter_length, **kwargs):
        self.supports_masking = True
        super(Convolution2D, self).__init__(nb_filter, filter_length, **kwargs)


class MaxPooling1D(native.MaxPooling1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaxPooling1D, self).__init__(**kwargs)


class Upsampling1D(native.UpSampling1D):
    def __init__(self, size, **kwargs):
        self.supports_masking = True
        super(Upsampling1D, self).__init__(size, **kwargs)


class Flatten(native.Flatten):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Flatten, self).__init__(**kwargs)


class Reshape(native.Reshape):
    def __init__(self, target_shape, **kwargs):
        self.supports_masking = True
        super(Reshape, self).__init__(target_shape, **kwargs)


class Deconvolution1D(_Conv):
    def __init__(self, bound_conv_layer=None,
                 filters=None,
                 kernel_size=None,
                 apply_mask=False,
                 strides=1,
                 padding=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 data_format='channels_last',
                 rank=1,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        self.supports_masking = True
        self.apply_mask = apply_mask
        if bound_conv_layer:
            self._bound_conv_layer = bound_conv_layer
            try:
                filters = self._bound_conv_layer.input_shape[2]
            except ValueError:
                filters = 'Not sure yet, input shape of convolutional layer not provided during construction.'
            kernel_size = self._bound_conv_layer.kernel_size
            padding = self._bound_conv_layer.padding

        super(Deconvolution1D, self).__init__(rank=rank,
                                              filters=filters,
                                              kernel_size=kernel_size,
                                              strides=strides,
                                              padding=padding,
                                              data_format=data_format,
                                              dilation_rate=dilation_rate,
                                              activation=activation,
                                              use_bias=use_bias,
                                              kernel_initializer=kernel_initializer,
                                              bias_initializer=bias_initializer,
                                              kernel_regularizer=kernel_regularizer,
                                              bias_regularizer=bias_regularizer,
                                              activity_regularizer=activity_regularizer,
                                              kernel_constraint=kernel_constraint,
                                              bias_constraint=bias_constraint,
                                              **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]

        if hasattr(self, '_bound_conv_layer'):
            self.filters = self._bound_conv_layer.input_shape[2]
            self.kernel = K.permute_dimensions(self._bound_conv_layer.kernel, (0, 2, 1))
        else:
            # This is a hack for loading through "model_from_json". It needs a fix.
            # For model loading, build first the arch and then load parameters.
            kernel_shape = self.kernel_size + (input_dim, self.filters)

            self.kernel = K.zeros(kernel_shape)

        if self.use_bias:
            self.bias = self.add_weight((self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, mask=None):

        outputs = K.conv1d(inputs,
                           self.kernel,
                           strides=self.strides[0],
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate[0])
        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            outputs = self.activation(outputs)

        # To do in the last only
        if self.apply_mask:
            outputs = outputs * K.cast(mask, K.floatx())
        return outputs


class Dedense(native.Dense):
    def __init__(self, bound_dense_layer=None,
                 units=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        if bound_dense_layer:
            self._bound_dense_layer = bound_dense_layer

            try:
                units = self._bound_dense_layer.input_shape[0]
            except IndexError:
                units = 'Not sure yet, input shape of dense layer not provided during construction.'

        super(Dedense, self).__init__(units,
                                      activation=activation,
                                      use_bias=use_bias,
                                      kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer,
                                      kernel_regularizer=kernel_regularizer,
                                      bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer,
                                      kernel_constraint=kernel_constraint,
                                      bias_constraint=bias_constraint,
                                      **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if hasattr(self, '_bound_dense_layer'):
            self.units = self._bound_dense_layer.input_shape[1]
            self.kernel = K.transpose(self._bound_dense_layer.kernel)
        else:
            # This is a hack for loading through "model_from_json". It needs a fix.
            # For model loading, build first the arch and then load parameters.
            self.kernel = K.zeros((input_dim, self.units))

        if self.use_bias:
            self.bias = self.add_weight((self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True


class FeedForwardLSTM(Recurrent):
    """Long-Short Term Memory unit - Hochreiter 1997.

    For a step-by-step description of the algorithm, see
    [this tutorial](http://deeplearning.net/tutorial/lstm.html).

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializers.get(init)
        self.inner_init = initializers.get(inner_init)
        self.forget_bias_init = initializers.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(FeedForwardLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        if self.consume_less == 'gpu':
            self.W = self.add_weight((self.input_dim, 4 * self.output_dim),
                                     initializer=self.init,
                                     name='{}_W'.format(self.name),
                                     regularizer=self.W_regularizer)
            self.U = self.add_weight((self.output_dim, 4 * self.output_dim),
                                     initializer=self.inner_init,
                                     name='{}_U'.format(self.name),
                                     regularizer=self.U_regularizer)

            def b_reg(shape, name=None):
                return K.variable(np.hstack((np.zeros(self.output_dim),
                                             K.get_value(self.forget_bias_init((self.output_dim,))),
                                             np.zeros(self.output_dim),
                                             np.zeros(self.output_dim))),
                                  name='{}_b'.format(self.name))

            self.b = self.add_weight((self.output_dim * 4,),
                                     initializer=b_reg,
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer)
        else:
            self.W_i = self.add_weight((self.input_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_W_i'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.U_i = self.add_weight((self.output_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_U_i'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.b_i = self.add_weight((self.output_dim,),
                                       initializer='zero',
                                       name='{}_b_i'.format(self.name),
                                       regularizer=self.b_regularizer)
            self.W_f = self.add_weight((self.input_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_W_f'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.U_f = self.add_weight((self.output_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_U_f'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.b_f = self.add_weight((self.output_dim,),
                                       initializer=self.forget_bias_init,
                                       name='{}_b_f'.format(self.name),
                                       regularizer=self.b_regularizer)
            self.W_c = self.add_weight((self.input_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_W_c'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.U_c = self.add_weight((self.output_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_U_c'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.b_c = self.add_weight((self.output_dim,),
                                       initializer='zero',
                                       name='{}_b_c'.format(self.name),
                                       regularizer=self.b_regularizer)
            self.W_o = self.add_weight((self.input_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_W_o'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.U_o = self.add_weight((self.output_dim, self.output_dim),
                                       initializer=self.init,
                                       name='{}_U_o'.format(self.name),
                                       regularizer=self.W_regularizer)
            self.b_o = self.add_weight((self.output_dim,),
                                       initializer='zero',
                                       name='{}_b_o'.format(self.name),
                                       regularizer=self.b_regularizer)

            self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                      self.W_c, self.U_c, self.b_c,
                                      self.W_f, self.U_f, self.b_f,
                                      self.W_o, self.U_o, self.b_o]
            self.W = K.concatenate([self.W_i, self.W_f, self.W_c, self.W_o])
            self.U = K.concatenate([self.U_i, self.U_f, self.U_c, self.U_o])
            self.b = K.concatenate([self.b_i, self.b_f, self.b_c, self.b_o])

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, a complete ' +
                             'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            if 0 < self.dropout_W < 1:
                dropout = self.dropout_W
            else:
                dropout = 0
            input_shape = K.int_shape(x)
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_i = time_distributed_dense(x, self.W_i, self.b_i, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_f = time_distributed_dense(x, self.W_f, self.b_f, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_c = time_distributed_dense(x, self.W_c, self.b_c, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_o = time_distributed_dense(x, self.W_o, self.b_o, dropout,
                                         input_dim, self.output_dim, timesteps)
            return K.concatenate([x_i, x_f, x_c, x_o], axis=2)
        else:
            return x

    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]

        if self.consume_less == 'gpu':
            z = K.dot(x * B_W[0], self.W) + K.dot(h_tm1 * B_U[0], self.U) + self.b

            z0 = z[:, :self.output_dim]
            z1 = z[:, self.output_dim: 2 * self.output_dim]
            z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
            z3 = z[:, 3 * self.output_dim:]

            i = self.inner_activation(z0)
            f = self.inner_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.inner_activation(z3)
        else:
            if self.consume_less == 'cpu':
                x_i = x[:, :self.output_dim]
                x_f = x[:, self.output_dim: 2 * self.output_dim]
                x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
                x_o = x[:, 3 * self.output_dim:]
            elif self.consume_less == 'mem':
                x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
                x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
                x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
                x_o = K.dot(x * B_W[3], self.W_o) + self.b_o
            else:
                raise ValueError('Unknown `consume_less` mode.')

            i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
            f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
            o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))

        h = o * self.activation(c)
        h_rec = h + self.inner_net(h)
        return h, [h_rec, c]

    def inner_net(self, x):
        inner_net_W1 = glorot_uniform((self.output_dim, self.output_dim))
        b1 = K.zeros((self.output_dim,))
        y = K.dot(x, inner_net_W1) + b1

        inner_net_W2 = glorot_uniform((self.output_dim, self.output_dim))
        b2 = K.zeros((self.output_dim,))
        output = K.dot(y, inner_net_W2) + b2

        return relu(output)

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.dropout_W < 1:
            input_shape = K.int_shape(x)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'forget_bias_init': self.forget_bias_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(FeedForwardLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



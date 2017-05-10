# -*- coding: utf-8 -*-
import functools
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import regularizers, activations, constraints, initializers
from keras.activations import relu
from keras.engine import InputSpec, Layer
from keras.initializers import glorot_uniform
from keras.layers.recurrent import Recurrent

import keras.layers as native
from keras.layers.convolutional import _Conv, Conv3D, conv_utils


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


class _SparseConv(Layer):
    """Abstract nD sparse convolution layer (private, used as implementation base).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    # Arguments
        rank: An integer, the rank of the convolution,
            e.g. "2" for 2D convolution.
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, ..., channels)` while `channels_first` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    """

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(_SparseConv, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = None

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            # self.bias = self.add_weight((self.filters,),
            #                             initializer=self.bias_initializer,
            #                             name='bias',
            #                             regularizer=self.bias_regularizer,
            #                             constraint=self.bias_constraint)
            raise NotImplementedError('bias in sparse conv is not implemented yet')
        else:
            self.bias = None
        # Set input spec.
        # self.input_spec = InputSpec(ndim=None,
        #                             axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        if self.rank == 1:
            # outputs = K.conv1d(
            #     inputs,
            #     self.kernel,
            #     strides=self.strides[0],
            #     padding=self.padding,
            #     data_format=self.data_format,
            #     dilation_rate=self.dilation_rate[0])
            raise NotImplementedError ('1D sparse conv not implemented yet')
        if self.rank == 2:
            # outputs = K.conv2d(
            #     inputs,
            #     self.kernel,
            #     strides=self.strides,
            #     padding=self.padding,
            #     data_format=self.data_format,
            #     dilation_rate=self.dilation_rate)
            raise NotImplementedError('2D sparse conv not implemented yet')
        if self.rank == 3:
            outputs = sparseConv3D(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
                activation=self.activation)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        # if self.activation is not None:
        #     return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            # space = input_shape[1:-1]
            # new_space = []
            # for i in range(len(space)):
            #     new_dim = conv_utils.conv_output_length(
            #         space[i],
            #         self.kernel_size[i],
            #         padding=self.padding,
            #         stride=self.strides[i],
            #         dilation=self.dilation_rate[i])
            #     new_space.append(new_dim)
            return (input_shape[0],) + tuple(None) + (self.filters,)
        if self.data_format == 'channels_first':
            # space = input_shape[2:]
            # new_space = []
            # for i in range(len(space)):
            #     new_dim = conv_utils.conv_output_length(
            #         space[i],
            #         self.kernel_size[i],
            #         padding=self.padding,
            #         stride=self.strides[i],
            #         dilation=self.dilation_rate[i])
            #     new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(None)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(_SparseConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SparseConv3D(_SparseConv):
    """3D sparse convolution layer (e.g. spatial convolution over volumes).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 128, 3)` for 128x128x128 volumes
    with a single channel,
    in `data_format="channels_last"`.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            width and height of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 3 integers,
            specifying the strides of the convolution along each spatial dimension.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `channels_first` corresponds to inputs with shape
            `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 3 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        5D tensor with shape:
        `(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if data_format='channels_first'
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if data_format='channels_last'.

    # Output shape
        5D tensor with shape:
        `(samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if data_format='channels_first'
        or 5D tensor with shape:
        `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` if data_format='channels_last'.
        `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have changed due to padding.
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1, 1),
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(SparseConv3D, self).__init__(
            rank=3,
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
        self.input_spec = None

    def get_config(self):
        config = super(SparseConv3D, self).get_config()
        config.pop('rank')
        return config


def sparseConv3D(x,
                 kernel,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 1, 1),
                 activation='linear'):
    """3D sparse convolution.

        # Arguments
            x: Tensor or variable.
            kernel: kernel tensor.
            strides: strides tuple.
            padding: string, `"same"` or `"valid"`.
            data_format: `"channels_last"` or `"channels_first"`.
                Whether to use Theano or TensorFlow data format
                for inputs/kernels/ouputs.
            dilation_rate: tuple of 3 integers.

        # Returns
            A sparse tensor, result of 3D convolution.

        # Raises
            ValueError: if `data_format` is neither `channels_last` or `channels_first`.
        """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))
    if data_format == 'channels_first':
        raise ValueError('channels first format is not supported for sparse conv')

    input_shape = x.dense_shape
    kernel_shape = K.int_shape(kernel)

    indices = x.indices
    values = tf.unstack(x.values)

    out_indices = []
    out_values = []

    for idx in tf.unstack(indices):
        list_idx = tf.unstack(idx)
        for i in range(max(0, int(list_idx[0]-(kernel_shape[0]-1)/dilation_rate[0])*dilation_rate[0]),
                       min(list_idx[0]+kernel_shape[0], input_shape[0]), dilation_rate[0]):
            for j in range(max(0, int(list_idx[1] - (kernel_shape[1] - 1)/dilation_rate[1])*dilation_rate[1]),
                           min(list_idx[1] + kernel_shape[1], input_shape[1]), dilation_rate[1]):
                for k in range(max(0, int(list_idx[2] - (kernel_shape[2] - 1)/dilation_rate[2])*dilation_rate[2]),
                               min(list_idx[2] + kernel_shape[2], input_shape[2]), dilation_rate[2]):

                    new_idx = tf.stack([i, j, k])
                    out_indices.append(new_idx)

                    local_indices = []
                    local_values = []
                    for l, idx_ in enumerate(tf.unstack(indices)):
                        if tf.reduce_all(tf.less_equal(tf.abs(tf.subtract(idx_, new_idx)),
                                                       tf.floor_div(tf.subtract(kernel_shape, 1), 2))):
                            local_indices.append(idx_)
                            local_values.append(values[l])

                    out_values.append(K.conv3d(K.to_dense(tf.SparseTensor(tf.stack(local_indices),
                                                                          tf.stack(local_values),
                                                                          kernel_shape)),
                                               kernel=kernel,
                                               strides=(1, 1, 1),
                                               data_format=data_format,
                                               dilation_rate=dilation_rate,
                                               padding='valid'))

    if padding == 'same':
        out_dense_shape = x.dense_shape
    else:
        out_dense_shape = [x.dense_shape[0]-(kernel_shape[0]-1),
                           x.dense_shape[1]-(kernel_shape[1]-1),
                           x.dense_shape[2]-(kernel_shape[2]-1)]

    return tf.SparseTensor(out_indices, out_values, out_dense_shape)


# def conv3d(x, kernel, strides=(1, 1, 1), padding='valid',
#            data_format=None, dilation_rate=(1, 1, 1)):
#     """3D convolution.
#
#     # Arguments
#         x: Tensor or variable.
#         kernel: kernel tensor.
#         strides: strides tuple.
#         padding: string, `"same"` or `"valid"`.
#         data_format: `"channels_last"` or `"channels_first"`.
#             Whether to use Theano or TensorFlow data format
#             for inputs/kernels/ouputs.
#         dilation_rate: tuple of 3 integers.
#
#     # Returns
#         A tensor, result of 3D convolution.
#
#     # Raises
#         ValueError: if `data_format` is neither `channels_last` or `channels_first`.
#     """
#     if data_format not in {'channels_first', 'channels_last'}:
#         raise ValueError('Unknown data_format ' + str(data_format))
#
#     # With 5d inputs, tf.nn.convolution only supports
#     # data_format NDHWC, so we transpose the inputs
#     # in case we are in data_format channels_first.
#     x = _preprocess_conv3d_input(x, data_format)
#     padding = _preprocess_padding(padding)
#     x = tf.nn.convolution(
#         input=x,
#         filter=kernel,
#         dilation_rate=dilation_rate,
#         strides=strides,
#         padding=padding,
#         data_format='NDHWC')
#     return _postprocess_conv3d_output(x, data_format)
custom_layers = {
    'Convolution1D': Convolution1D,
    'Deconvolution1D': Deconvolution1D,
    'Convolution2D': Convolution2D,
    'MaxPooling1D': MaxPooling1D,
    'Upsampling1D': Upsampling1D,
    'Dedense': Dedense,
    'Reshape': Reshape,
    'Flatten': Flatten,
    'LocallyConnected1D': LocallyConnected1D,
    'FeedForwardLSTM': FeedForwardLSTM,
    'SparseConv3D': SparseConv3D
    # 'AtrousConv1D': AtrousConv1D
}

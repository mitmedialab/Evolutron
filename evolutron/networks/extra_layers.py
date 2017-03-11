# -*- coding: utf-8 -*-
import functools
import keras.backend as K
import numpy as np
from keras.layers import regularizers, initializations, activations, constraints
from keras.activations import relu
from keras.engine import InputSpec
from keras.initializations import glorot_uniform
from keras.layers import LocallyConnected1D
from keras.layers.pooling import _Pooling1D
from keras.layers.recurrent import Recurrent, time_distributed_dense
from keras.models import Layer
from keras.utils.np_utils import conv_output_length

import keras.layers as native


class Convolution1D(native.Convolution1D):
    def __init__(self, nb_filter, filter_length, **kwargs):
        self.supports_masking = True
        super(Convolution1D, self).__init__(nb_filter, filter_length, **kwargs)


class Dense(native.Dense):
    def __init__(self, output_dim, **kwargs):
        self.supports_masking = True
        super(Dense, self).__init__(output_dim, **kwargs)


class MaxPooling1D(native.MaxPooling1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaxPooling1D, self).__init__(**kwargs)


class Flatten(native.Flatten):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Flatten, self).__init__(**kwargs)


class Reshape(native.Reshape):
    def __init__(self, target_shape, **kwargs):
        self.supports_masking = True
        super(Reshape, self).__init__(target_shape, **kwargs)


class Deconvolution1D(Layer):
    def __init__(self, bound_conv_layer=None, nb_filter=None, filter_length=None, apply_mask=False,
                 init='uniform', activation='linear', weights=None, subsample_length=1,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, border_mode='valid',
                 bias=True, input_dim=None, input_length=None, **kwargs):

        self.supports_masking = True
        self.apply_mask = apply_mask

        if 'border_mode' in kwargs:
            raise Exception('Border mode is inferred from Conv Layer')

        if bound_conv_layer:  # If instanciated through a connection
            self._bound_conv_layer = bound_conv_layer
            try:
                self.nb_filter = self._bound_conv_layer.input_shape[2]
            except ValueError:
                self.nb_filter = 'Not sure yet, input shape of convolutional layer not provided during construction.'
            self.filter_length = self._bound_conv_layer.filter_length
            self.border_mode = self._bound_conv_layer.border_mode
        else:
            # if instanciated through config
            self.nb_filter = nb_filter
            self.filter_length = filter_length
            self.border_mode = border_mode

        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.subsample_length = subsample_length

        self.subsample = (subsample_length, 1)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.input_spec = [InputSpec(ndim=3)]
        self.initial_weights = weights
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(Deconvolution1D, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[2]

        if hasattr(self, '_bound_conv_layer'):
            self.nb_filter = self._bound_conv_layer.input_shape[2]
            self.W_shape = (self.filter_length, 1, input_dim, self.nb_filter)
            self.W = K.permute_dimensions(self._bound_conv_layer.W, (0, 1, 3, 2))
        else:
            self.W_shape = (self.filter_length, 1, input_dim, self.nb_filter)
            self.W = self.add_weight(self.W_shape,
                                     initializer=functools.partial(self.init, dim_ordering='th'),
                                     name='{}_W'.format(self.name),
                                     regularizer=self.W_regularizer,
                                     constraint=self.W_constraint)

        if self.bias:
            self.b = self.add_weight((self.nb_filter,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def get_output_shape_for(self, input_shape):
        length = conv_output_length(input_shape[1],
                                    self.filter_length,
                                    self.border_mode,
                                    self.subsample[0])
        return input_shape[0], length, self.nb_filter

    def call(self, x, mask=None):
        x = K.expand_dims(x, 2)  # add a dummy dimension
        output = K.conv2d(x, self.W, strides=self.subsample,
                          border_mode=self.border_mode,
                          dim_ordering='tf')
        output = K.squeeze(output, 2)  # remove the dummy dimension
        if self.bias:
            output += K.reshape(self.b, (1, 1, self.nb_filter))

        output = self.activation(output)
        # To do in the last only
        if self.apply_mask:
            output = output * K.cast(mask, K.floatx())
        return output

    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'filter_length': self.filter_length,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample_length': self.subsample_length,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(Deconvolution1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# class AtrousConvolution1D(Convolution1D):
#     '''Atrous Convolution operator for filtering neighborhoods of one-dimensional inputs.
#     A.k.a dilated convolution or convolution with holes.
#     When using this layer as the first layer in a model,
#     either provide the keyword argument `input_dim`
#     (int, e.g. 128 for sequences of 128-dimensional vectors),
#     or `input_shape` (tuples of integers, e.g. (10, 128) for sequences
#     of 10 vectors of 128-dimensional vectors).
#
#     # Example
#
#     ```python
#         # apply an atrous convolution 1d with atrous rate 2 of length 3 to a sequence with 10 timesteps,
#         # with 64 output filters
#         model = Sequential()
#         model.add(AtrousConvolution1D(64, 3, atrous_rate=2, border_mode='same', input_shape=(10, 32)))
#         # now model.output_shape == (None, 10, 64)
#
#         # add a new atrous conv1d on top
#         model.add(AtrousConvolution1D(32, 3, atrous_rate=2, border_mode='same'))
#         # now model.output_shape == (None, 10, 32)
#     ```
#
#     # Arguments
#         nb_filter: Number of convolution kernels to use
#             (dimensionality of the output).
#         filter_length: The extension (spatial or temporal) of each filter.
#         init: name of initialization function for the weights of the layer
#             (see [initializations](../initializations.md)),
#             or alternatively, Theano function to use for weights initialization.
#             This parameter is only relevant if you don't pass a `weights` argument.
#         activation: name of activation function to use
#             (see [activations](../activations.md)),
#             or alternatively, elementwise Theano function.
#             If you don't specify anything, no activation is applied
#             (ie. "linear" activation: a(x) = x).
#         weights: list of numpy arrays to set as initial weights.
#         border_mode: 'valid' or 'same'.
#         subsample_length: factor by which to subsample output.
#         atrous_rate: Factor for kernel dilation. Also called filter_dilation
#             elsewhere.
#         W_regularizer: instance of [WeightRegularizer](../regularizers.md)
#             (eg. L1 or L2 regularization), applied to the main weights matrix.
#         b_regularizer: instance of [WeightRegularizer](../regularizers.md),
#             applied to the bias.
#         activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
#             applied to the network output.
#         W_constraint: instance of the [constraints](../constraints.md) module
#             (eg. maxnorm, nonneg), applied to the main weights matrix.
#         b_constraint: instance of the [constraints](../constraints.md) module,
#             applied to the bias.
#         bias: whether to include a bias
#             (i.e. make the layer affine rather than linear).
#         input_dim: Number of channels/dimensions in the input.
#             Either this argument or the keyword argument `input_shape`must be
#             provided when using this layer as the first layer in a model.
#         input_length: Length of input sequences, when it is constant.
#             This argument is required if you are going to connect
#             `Flatten` then `Dense` layers upstream
#             (without it, the shape of the dense outputs cannot be computed).
#
#     # Input shape
#         3D tensor with shape: `(samples, steps, input_dim)`.
#
#     # Output shape
#         3D tensor with shape: `(samples, new_steps, nb_filter)`.
#         `steps` value might have changed due to padding.
#     '''
#
#     def __init__(self, nb_filter, filter_length,
#                  init='uniform', activation='linear', weights=None,
#                  border_mode='valid', subsample_length=1, atrous_rate=1,
#                  W_regularizer=None, b_regularizer=None, activity_regularizer=None,
#                  W_constraint=None, b_constraint=None,
#                  bias=True, **kwargs):
#
#         self.supports_masking = True
#
#         if border_mode not in {'valid', 'same'}:
#             raise Exception('Invalid border mode for AtrousConv1D:', border_mode)
#
#         self.atrous_rate = int(atrous_rate)
#
#         super(AtrousConvolution1D, self).__init__(nb_filter, filter_length,
#                                                   init=init, activation=activation,
#                                                   weights=weights, border_mode=border_mode,
#                                                   subsample_length=subsample_length,
#                                                   W_regularizer=W_regularizer, b_regularizer=b_regularizer,
#                                                   activity_regularizer=activity_regularizer,
#                                                   W_constraint=W_constraint, b_constraint=b_constraint,
#                                                   bias=bias, **kwargs)
#
#     def get_output_shape_for(self, input_shape):
#         length = conv_output_length(input_shape[1],
#                                     self.filter_length,
#                                     self.border_mode,
#                                     self.subsample[0],
#                                     dilation=self.atrous_rate)
#         return (input_shape[0], length, self.nb_filter)
#
#     def call(self, x, mask=None):
#         x = K.expand_dims(x, 2)  # add a dummy dimension
#         output = K.conv2d(x, self.W, strides=self.subsample,
#                           border_mode=self.border_mode,
#                           dim_ordering='tf',
#                           filter_dilation=(self.atrous_rate, self.atrous_rate))
#         output = K.squeeze(output, 2)  # remove the dummy dimension
#         if self.bias:
#             output += K.reshape(self.b, (1, 1, self.nb_filter))
#         output = self.activation(output)
#         return output
#
#     def get_config(self):
#         config = {'atrous_rate': self.atrous_rate}
#         base_config = super(AtrousConvolution1D, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#
# class Convolution2D(Layer):
#     '''Convolution operator for filtering windows of two-dimensional inputs.
#     When using this layer as the first layer in a model,
#     provide the keyword argument `input_shape`
#     (tuple of integers, does not include the sample axis),
#     e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.
#
#     # Examples
#
#     ```python
#         # apply a 3x3 convolution with 64 output filters on a 256x256 image:
#         model = Sequential()
#         model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 256, 256)))
#         # now model.output_shape == (None, 64, 256, 256)
#
#         # add a 3x3 convolution on top, with 32 output filters:
#         model.add(Convolution2D(32, 3, 3, border_mode='same'))
#         # now model.output_shape == (None, 32, 256, 256)
#     ```
#
#     # Arguments
#         nb_filter: Number of convolution filters to use.
#         nb_row: Number of rows in the convolution kernel.
#         nb_col: Number of columns in the convolution kernel.
#         init: name of initialization function for the weights of the layer
#             (see [initializations](../initializations.md)), or alternatively,
#             Theano function to use for weights initialization.
#             This parameter is only relevant if you don't pass
#             a `weights` argument.
#         activation: name of activation function to use
#             (see [activations](../activations.md)),
#             or alternatively, elementwise Theano function.
#             If you don't specify anything, no activation is applied
#             (ie. "linear" activation: a(x) = x).
#         weights: list of numpy arrays to set as initial weights.
#         border_mode: 'valid' or 'same'.
#         subsample: tuple of length 2. Factor by which to subsample output.
#             Also called strides elsewhere.
#         W_regularizer: instance of [WeightRegularizer](../regularizers.md)
#             (eg. L1 or L2 regularization), applied to the main weights matrix.
#         b_regularizer: instance of [WeightRegularizer](../regularizers.md),
#             applied to the bias.
#         activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
#             applied to the network output.
#         W_constraint: instance of the [constraints](../constraints.md) module
#             (eg. maxnorm, nonneg), applied to the main weights matrix.
#         b_constraint: instance of the [constraints](../constraints.md) module,
#             applied to the bias.
#         dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
#             (the depth) is at index 1, in 'tf' mode is it at index 3.
#             It defaults to the `image_dim_ordering` value found in your
#             Keras config file at `~/.keras/keras.json`.
#             If you never set it, then it will be "th".
#         bias: whether to include a bias
#             (i.e. make the layer affine rather than linear).
#
#     # Input shape
#         4D tensor with shape:
#         `(samples, channels, rows, cols)` if dim_ordering='th'
#         or 4D tensor with shape:
#         `(samples, rows, cols, channels)` if dim_ordering='tf'.
#
#     # Output shape
#         4D tensor with shape:
#         `(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
#         or 4D tensor with shape:
#         `(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
#         `rows` and `cols` values might have changed due to padding.
#     '''
#
#     def __init__(self, nb_filter, nb_row, nb_col,
#                  init='glorot_uniform', activation='linear', weights=None,
#                  border_mode='valid', subsample=(1, 1), dim_ordering='default',
#                  W_regularizer=None, b_regularizer=None, activity_regularizer=None,
#                  W_constraint=None, b_constraint=None,
#                  bias=True, **kwargs):
#
#         self.supports_masking = True
#
#         if dim_ordering == 'default':
#             dim_ordering = K.image_dim_ordering()
#         if border_mode not in {'valid', 'same'}:
#             raise Exception('Invalid border mode for Convolution2D:', border_mode)
#         self.nb_filter = nb_filter
#         self.nb_row = nb_row
#         self.nb_col = nb_col
#         self.init = initializations.get(init, dim_ordering=dim_ordering)
#         self.activation = activations.get(activation)
#         assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
#         self.border_mode = border_mode
#         self.subsample = tuple(subsample)
#         assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
#         self.dim_ordering = dim_ordering
#
#         self.W_regularizer = regularizers.get(W_regularizer)
#         self.b_regularizer = regularizers.get(b_regularizer)
#         self.activity_regularizer = regularizers.get(activity_regularizer)
#
#         self.W_constraint = constraints.get(W_constraint)
#         self.b_constraint = constraints.get(b_constraint)
#
#         self.bias = bias
#         self.input_spec = [InputSpec(ndim=4)]
#         self.initial_weights = weights
#         super(Convolution2D, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         if self.dim_ordering == 'th':
#             stack_size = input_shape[1]
#             self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
#         elif self.dim_ordering == 'tf':
#             stack_size = input_shape[3]
#             self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
#         else:
#             raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
#         self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
#         if self.bias:
#             self.b = K.zeros((self.nb_filter,), name='{}_b'.format(self.name))
#             self.trainable_weights = [self.W, self.b]
#         else:
#             self.trainable_weights = [self.W]
#         self.regularizers = []
#
#         if self.W_regularizer:
#             self.W_regularizer.set_param(self.W)
#             self.regularizers.append(self.W_regularizer)
#
#         if self.bias and self.b_regularizer:
#             self.b_regularizer.set_param(self.b)
#             self.regularizers.append(self.b_regularizer)
#
#         if self.activity_regularizer:
#             self.activity_regularizer.set_layer(self)
#             self.regularizers.append(self.activity_regularizer)
#
#         self.constraints = {}
#         if self.W_constraint:
#             self.constraints[self.W] = self.W_constraint
#         if self.bias and self.b_constraint:
#             self.constraints[self.b] = self.b_constraint
#
#         if self.initial_weights is not None:
#             self.set_weights(self.initial_weights)
#             del self.initial_weights
#
#     def get_output_shape_for(self, input_shape):
#         if self.dim_ordering == 'th':
#             rows = input_shape[2]
#             cols = input_shape[3]
#         elif self.dim_ordering == 'tf':
#             rows = input_shape[1]
#             cols = input_shape[2]
#         else:
#             raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
#
#         rows = conv_output_length(rows, self.nb_row,
#                                   self.border_mode, self.subsample[0])
#         cols = conv_output_length(cols, self.nb_col,
#                                   self.border_mode, self.subsample[1])
#
#         if self.dim_ordering == 'th':
#             return (input_shape[0], self.nb_filter, rows, cols)
#         elif self.dim_ordering == 'tf':
#             return (input_shape[0], rows, cols, self.nb_filter)
#         else:
#             raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
#
#     def call(self, x, mask=None):
#         output = K.conv2d(x, self.W, strides=self.subsample,
#                           border_mode=self.border_mode,
#                           dim_ordering=self.dim_ordering,
#                           filter_shape=self.W_shape)
#         if self.bias:
#             if self.dim_ordering == 'th':
#                 output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
#             elif self.dim_ordering == 'tf':
#                 output += K.reshape(self.b, (1, 1, 1, self.nb_filter))
#             else:
#                 raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
#         output = self.activation(output)
#         return output
#
#     def get_config(self):
#         config = {'nb_filter': self.nb_filter,
#                   'nb_row': self.nb_row,
#                   'nb_col': self.nb_col,
#                   'init': self.init.__name__,
#                   'activation': self.activation.__name__,
#                   'border_mode': self.border_mode,
#                   'subsample': self.subsample,
#                   'dim_ordering': self.dim_ordering,
#                   'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
#                   'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
#                   'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
#                   'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
#                   'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
#                   'bias': self.bias}
#         base_config = super(Convolution2D, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#


class Dedense(Layer):
    def __init__(self, bound_dense_layer=None, output_dim=None, init='glorot_uniform',
                 activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, **kwargs):

        self.supports_masking = True

        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        if bound_dense_layer:
            self._bound_dense_layer = bound_dense_layer

            try:
                self.output_dim = self._bound_dense_layer.input_shape[0]
            except Exception:
                self.output_dim = 'Not sure yet, input shape of dense layer not provided during construction.'
        else:
            self.output_dim = output_dim

        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim='2+')]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Dedense, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        if hasattr(self, '_bound_dense_layer'):
            self.output_dim = self._bound_dense_layer.input_shape[1]
            self.W = K.transpose(self._bound_dense_layer.W)
        else:
            self.W = self.add_weight((input_dim, self.output_dim),
                                     initializer=self.init,
                                     name='{}_W'.format(self.name),
                                     regularizer=self.W_regularizer,
                                     constraint=self.W_constraint)

        if self.bias:
            self.b = self.add_weight((self.output_dim,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        output = K.dot(x, self.W)
        if self.bias:
            output += self.b
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(Dedense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Unpooling1D(Layer):
    """Repeat each temporal step `length` times along the time axis.

    # Arguments
        length: integer. Upsampling factor.

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.

    # Output shape
        3D tensor with shape: `(samples, upsampled_steps, features)`.
    """

    def __init__(self, bound_pool_layer=None, length=None, **kwargs):

        self.supports_masking = True

        if bound_pool_layer:
            self._bound_pool_layer = bound_pool_layer
            try:
                self.length = self._bound_pool_layer.input_shape[1]
            except Exception:
                self.length = 'Not sure yet, input shape not provided during construction.'
        else:
            self.length = length

        self.input_spec = [InputSpec(ndim=3)]
        super(Unpooling1D, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        length = self.length * input_shape[1] if input_shape[1] and self.length is not None else None
        return (input_shape[0], length, input_shape[2])

    def call(self, x, mask=None):

        if hasattr(self, '_bound_pool_layer'):
            self.length = self._bound_pool_layer.input_shape[1]

        pre_output = K.repeat_elements(x, self.length, axis=1)

        if hasattr(self,
                   '_bound_pool_layer') and K.backend() == 'theano':  # if we have theano backend, we get nice upsample
            # TODO: implement this when loading
            import theano.tensor as T
            output = T.grad(K.sum(self._bound_pool_layer.output), wrt=self._bound_pool_layer.input) * pre_output

        else:
            # TODO: implement unpooling in tensorflow
            output = pre_output

        return output

    def get_config(self):
        config = {'length': self.length}
        base_config = super(Unpooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class feedForwardLSTM(Recurrent):
    '''Long-Short Term Memory unit - Hochreiter 1997.

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
    '''

    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(feedForwardLSTM, self).__init__(**kwargs)

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
        base_config = super(feedForwardLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LocallyConnected1D(Layer):
    '''The `LocallyConnected1D` layer works similarly to
    the `Convolution1D` layer, except that weights are unshared,
    that is, a different set of filters is applied at each different patch
    of the input.
    When using this layer as the first layer in a model,
    either provide the keyword argument `input_dim`
    (int, e.g. 128 for sequences of 128-dimensional vectors), or `input_shape`
    (tuple of integers, e.g. `input_shape=(10, 128)`
    for sequences of 10 vectors of 128-dimensional vectors).
    Also, note that this layer can only be used with
    a fully-specified input shape (`None` dimensions not allowed).

    # Example
    ```python
        # apply a unshared weight convolution 1d of length 3 to a sequence with
        # 10 timesteps, with 64 output filters
        model = Sequential()
        model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
        # now model.output_shape == (None, 8, 64)
        # add a new conv1d on top
        model.add(LocallyConnected1D(32, 3))
        # now model.output_shape == (None, 6, 32)
    ```

    # Arguments
        nb_filter: Dimensionality of the output.
        filter_length: The extension (spatial or temporal) of each filter.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: Only support 'valid'. Please make good use of
            ZeroPadding1D to achieve same output length.
        subsample_length: factor by which to subsample output.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias (i.e. make the layer affine rather than linear).
        input_dim: Number of channels/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.
        input_length: Length of input sequences, when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).

    # Input shape
        3D tensor with shape: `(samples, steps, input_dim)`.

    # Output shape
        3D tensor with shape: `(samples, new_steps, nb_filter)`.
        `steps` value might have changed due to padding.
    '''

    def __init__(self, nb_filter, filter_length,
                 init='glorot_uniform', activation=None, weights=None,
                 border_mode='valid', subsample_length=1,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, input_length=None, **kwargs):
        self.supports_masking = True

        if border_mode != 'valid':
            raise ValueError('Invalid border mode for LocallyConnected1D '
                             '(only "valid" is supported):', border_mode)
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.init = initializations.get(init, dim_ordering='th')
        self.activation = activations.get(activation)

        self.border_mode = border_mode
        self.subsample_length = subsample_length

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.input_spec = [InputSpec(ndim=3)]
        self.initial_weights = weights
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(LocallyConnected1D, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[2]
        _, output_length, nb_filter = self.get_output_shape_for(input_shape)
        self.W_shape = (output_length,
                        self.filter_length * input_dim,
                        nb_filter)
        # self.W_shape = (nb_filter, input_shape[1], input_dim)
        self.W = self.add_weight(self.W_shape,
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((output_length, self.nb_filter),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def get_output_shape_for(self, input_shape):
        length = conv_output_length(input_shape[1],
                                    self.filter_length,
                                    self.border_mode,
                                    self.subsample_length)
        return (input_shape[0], length, self.nb_filter)

    """def call(self, x, mask=None):
        stride = self.subsample_length
        assert stride == 1, 'locally connected only supports 1 stride and 1 length'
        nb_filter, _, _ = self.W_shape

        #if K._backend == 'theano':
        x = x.reshape([x.shape[0], 1, x.shape[1], x.shape[2]])
        output = K.sum(x * self.W, axis=3)  # uses broadcasting, sums over input filters
        return output"""

    def call(self, x, mask=None):
        stride = self.subsample_length
        output_length, feature_dim, nb_filter = self.W_shape

        xs = []
        for i in range(output_length):
            slice_length = slice(i * stride, i * stride + self.filter_length)
            xs.append(K.reshape(x[:, slice_length, :], (1, -1, feature_dim)))
        x_aggregate = K.concatenate(xs, axis=0)
        # (output_length, batch_size, nb_filter)
        output = K.batch_dot(x_aggregate, self.W)
        output = K.permute_dimensions(output, (1, 0, 2))

        if self.bias:
            output += K.reshape(self.b, (1, output_length, nb_filter))

        output = self.activation(output)
        return output

    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'filter_length': self.filter_length,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample_length': self.subsample_length,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(LocallyConnected1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

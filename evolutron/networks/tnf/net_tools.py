# coding=utf-8
import tensorflow as tf


def weight_variable(shape=None, initializer=tf.contrib.layers.xavier_initializer(), W_relu=False, name=None):
    with tf.variable_scope(name):
        initial = tf.get_variable('initial_weight', shape, dtype=tf.float32, initializer=initializer)
    if W_relu:
        return tf.nn.relu(initial)
    else:
        return initial


def bias_variable(shape=None, initializer=tf.constant_initializer(), name=None):
    with tf.variable_scope(name):
        initial = tf.get_variable('initial_bias', shape, dtype=tf.float32, initializer=initializer)
    return initial


def FC_layer(x, num_units, nonlinearity=tf.nn.relu,
             W=tf.contrib.layers.xavier_initializer(),
             b=tf.constant_initializer(), W_relu=False, name=None):
    w = weight_variable(shape=[(x.get_shape().as_list())[-1], num_units], initializer=W, W_relu=W_relu, name=name)
    b = bias_variable(shape=[num_units], initializer=b, name=name)

    variable_summaries(w, name + '/weights')
    variable_summaries(b, name + '/biases')

    return nonlinearity(tf.matmul(x, w) + b)


def conv2d_layer(x, num_filters, filter_size, stride=1, pad='VALID',
                 nonlinearity=tf.nn.relu,
                 W=tf.contrib.layers.xavier_initializer(),
                 b=tf.constant_initializer(), W_relu=False, name=None):
    x_shape = x.get_shape().as_list()

    try:
        filter_size[1]
    except:
        filter_size = [filter_size, filter_size]

    w = weight_variable(shape=[filter_size[0], filter_size[1], x_shape[-1], num_filters], initializer=W, W_relu=W_relu,
                        name=name)

    b = bias_variable(shape=[num_filters], initializer=b, name=name)

    variable_summaries(w, name + '/weights')
    variable_summaries(b, name + '/biases')

    return nonlinearity(tf.nn.conv2d(x, w, stride, pad) + b)


def conv1d_layer(x, num_filters, filter_size, stride=1, pad='VALID',
                 nonlinearity=tf.nn.relu,
                 W=tf.contrib.layers.xavier_initializer(),
                 b=tf.constant_initializer(), W_relu=False, name=None):
    x = tf.transpose(x, [0, 2, 1])
    x = tf.expand_dims(x, -2)
    x_shape = x.get_shape().as_list()

    w = weight_variable(shape=[filter_size, 1, x_shape[-1], num_filters], initializer=W, W_relu=W_relu, name=name)
    b = bias_variable(shape=[num_filters], initializer=b, name=name)

    variable_summaries(w, name + '/weights')
    variable_summaries(b, name + '/biases')

    result = tf.squeeze(nonlinearity(tf.nn.conv2d(x, w, [1, 1, stride, 1], pad) + b), squeeze_dims=[2])
    return tf.transpose(result, [0, 2, 1])


def max_pool(x, pool_function=None, name=None, keep_dims=False):
    shape = x.get_shape().as_list()
    return tf.reduce_max(x, [i for i in range(2, len(shape))], keep_dims=keep_dims)


def cat_crossentropy(x, y):
    y = tf.reshape(y, x.shape)
    crossentropy = -tf.reduce_sum(x * tf.log(tf.clip_by_value(y, 1e-10, 10.0)), reduction_indices=[1])

    return crossentropy


def bin_crossentropy(x, y):
    x_shape = x.get_shape().as_list()
    for i in range(len(x_shape)):
        if x_shape[i] == None:
            x_shape[i] = -1

    y = tf.reshape(y, x_shape)
    crossentropy = -y * tf.log(tf.clip_by_value(x, 1e-10, 10.0)) - (1 - y) * tf.log(
        tf.clip_by_value(1 - x, 1e-10, 10.0))
    return crossentropy


def bin_accuracy(a, b, t=0.5):
    return tf.cast(tf.equal(tf.cast(a >= t, tf.float32), b), tf.float32)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

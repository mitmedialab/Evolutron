# coding=utf-8

from .krs.extra_layers import (Convolution1D, Deconvolution1D, Convolution2D,
                               MaxPooling1D, Upsampling1D,
                               Dedense, Reshape, Flatten,
                               LocallyConnected1D,
                               FeedForwardLSTM)

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
    # 'AtrousConv1D': AtrousConv1D
}

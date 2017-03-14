# coding=utf-8

from .krs.extra_layers import (Convolution1D, AtrousConvolution1D, Deconvolution1D, Convolution2D,
                               MaxPooling1D, Unpooling1D,
                               Dense, Dedense, Reshape, Flatten,
                               LocallyConnected1D,
                               FeedForwardLSTM)

custom_layers = {
    'Convolution1D': Convolution1D,
    'Deconvolution1D': Deconvolution1D,
    'Convolution2D': Convolution2D,
    'MaxPooling1D': MaxPooling1D,
    'Unpooling1D': Unpooling1D,
    'Dense': Dense,
    'Dedense': Dedense,
    'Reshape': Reshape,
    'Flatten': Flatten,
    'LocallyConnected1D': LocallyConnected1D,
    'FeedForwardLSTM': FeedForwardLSTM
}
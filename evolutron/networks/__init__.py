# coding=utf-8

from .krs.comet import DeepCoDER, DeepCoFAM

from .krs.extra_layers import Deconvolution1D, Dedense, Unpooling1D

from .krs import extra_layers as extra

custom_layers = {
    'Convolution1D': extra.Convolution1D,
    'Deconvolution1D': Deconvolution1D,
    'MaxPooling1D': extra.MaxPooling1D,
    'Unpooling1D': Unpooling1D,
    'Dense': extra.Dense,
    'Dedense': Dedense,
    'Reshape': extra.Reshape,
    'Flatten': extra.Flatten
}

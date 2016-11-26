# coding=utf-8

from .krs.comet import DeepCoDER, DeepCoFAM

from .krs.cobind import DeepDNABindN

from .krs import extra_layers as extra

custom_layers = {
    'Convolution1D': extra.Convolution1D,
    'Deconvolution1D': extra.Deconvolution1D,
    'MaxPooling1D': extra.MaxPooling1D,
    'Unpooling1D': extra.Unpooling1D,
    'Dense': extra.Dense,
    'Dedense': extra.Dedense,
    'Reshape': extra.Reshape,
    'Flatten': extra.Flatten
}

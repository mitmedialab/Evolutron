#!/usr/bin/env python
# coding=utf-8
"""
    SecS - Secondary Structure
    ------------------------------------------_
    SecStructure is an automated tool for prediction of protein secondary structure
     from it's amino acid sequence.

    (c) Massachusetts Institute of Technology

    For more information contact:
    kfirs [at] mit.edu
"""

import argparse
import time

import numpy as np

# Check if package is installed, else fallback to developer mode imports
try:
    import Structure.SecS as SecS
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.abspath('..'))
    import Structure.SecS as SecS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SecS - Secondary Structure Prediction Tool',
                                     argument_default=argparse.SUPPRESS)

    parser.add_argument("data_id", default='cullPDB',
                        help='The protein dataset to be trained on.')

    parser.add_argument("--filters", type=int,
                        help='Number of filters in the convolutional layers.')

    parser.add_argument("--filter_length", type=int,
                        help='Size of filters in the first convolutional layer.')

    parser.add_argument("--no_pad", action='store_true',
                        help='Toggle to pad protein sequences. Batch size auto-change to 1.')

    parser.add_argument("--conv", type=int,
                        help='number of conv layers.')

    parser.add_argument("--fc", type=int, default=0,
                        help='number of fc layers.')

    parser.add_argument("--lstm", type=int, default=1,
                        help='number of fc layers.')

    parser.add_argument("-e", "--epochs", default=1000, type=int,
                        help='number of training epochs to perform (default: 1000)')

    parser.add_argument("-b", "--batch_size", type=int, default=256,
                        help='Size of minibatch.')

    parser.add_argument("--rate", type=float,
                        help='The learning rate for the optimizer.')

    parser.add_argument("--clipnorm", type=float,
                        help='Clipping the gradient norm for the optimizer.')

    parser.add_argument("--model", type=str,
                        help='Continue training the given model. Other architecture options are unused.')

    parser.add_argument("--optimizer", choices=['adam', 'nadam', 'rmsprop', 'sgd', 'adadelta', 'adagrad'],
                        help='The optimizer to be used.')

    parser.add_argument('--nb_categories', '-c', type=int, default=8,
                        help='how many categories (3/8)?')

    parser.add_argument('--dilation', type=int, default=1,
                        help='dilation?')

    parser.add_argument('--validation', type=float, default=.05,
                        help='validation set size?')

    parser.add_argument("--mode", choices=['NaNGuardMode'],
                        help='Theano mode to be used.')

    args = parser.parse_args()

    kwargs = args.__dict__

    count = 0

    for dilation in [1, 3, 5, 10]:
        for clipnorm in [1, .5, .1]:
            #for conv in [5, 3, 9, 7, 1]:
                for filter_length in [30, 15, 9, 5, 50]:
                    for rate in [10 ** i for i in range(-2, 1, 1)]:
                        kwargs['rate'] = rate
                        #kwargs['conv'] = conv
                        kwargs['filter_length'] = filter_length
                        kwargs['dilation'] = dilation
                        kwargs['clipnorm'] = clipnorm

                        # sys.stdout = file
                        score, cb513_score, casp10_score, casp11_score, handle = SecS.main(**kwargs)
                        # sys.stdout = sys.__stdout__

                        with open('HyperParameterOpt.txt', mode='a') as file:
                            file.write('%d,%d,%d,%d,%.2f,%.2f\n'
                                       '%.3f, %.3f\n%.3f, %.3f\n'
                                       '%.3f, %.3f\n%.3f, %.3f\n' %
                                       (kwargs['lstm'], kwargs['conv'], filter_length, dilation, rate, clipnorm,
                                        score[0], 100 * score[1], cb513_score[0], 100 * cb513_score[1],
                                        casp10_score[0], 100 * casp10_score[1], casp11_score[0], 100 * casp11_score[1]))




                            count += 1
                            print('Finished %d iterations' % count)

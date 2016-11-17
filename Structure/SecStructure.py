from keras.layers import Input, LSTM, Activation, Dropout, Masking
try:
    from .extra_layers import Convolution1D, MaxPooling1D, Dense, Flatten, Reshape
except Exception: #ImportError
    from extra_layers import Convolution1D, MaxPooling1D, Dense, Flatten, Reshape
from keras.models import Model
from keras.optimizers import SGD, Nadam
from keras.regularizers import l2, activity_l1
#from keras.utils.visualize_util import model_to_dot
from keras.objectives import categorical_crossentropy
import keras.backend as K

from IPython.display import SVG

import numpy as np
import argparse

# Check if package is installed, else fallback to developer mode imports
try:
    from evolutron.tools import load_dataset
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.abspath('..'))

    from evolutron.tools import load_dataset


def SecStructure(args):
    dataset, secondary_structure = load_dataset(data_id, padded=True)

    num_examples = dataset.shape[0]
    example_len = dataset.shape[1]
    num_channels = dataset.shape[2]

    print(dataset.shape)

    input_aa_seq = Input(shape=(example_len, num_channels))

    masking = Masking(mask_value=0)(input_aa_seq)

    lstm = LSTM(output_dim=args.num_categories, input_shape=(None, example_len, num_channels),
                return_sequences=True, W_regularizer=l2(0.1))(masking)

    flat = Flatten()(lstm)

    dropout = Dropout(p=0.3)(flat)

    dense = Dense(output_dim=example_len*args.num_categories,
                  activity_regularizer=activity_l1(0.1))(dropout)

    reshape = Reshape(target_shape=(example_len, args.num_categories))(dense)

    output = Activation(activation='softmax')(reshape)

    model = Model(input=input_aa_seq, output=output)

    sgd = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

    nadam = Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    model.compile(optimizer=nadam, loss=vector_categorical_crossentropy, metrics=['categorical_accuracy'])

    model.summary()

    model.fit(dataset, secondary_structure,
              batch_size=args.batch_size,
              nb_epoch=args.num_epochs,
              shuffle=True,
              validation_split=.2)

    model.save('tmp')

    #SVG(model_to_dot(model).create(prog='dot', format='svg'))

    print(dataset[0:2, :, :])
    print(secondary_structure[0:2, :, :])
    print(model.predict(dataset[0:2, :, :]))


# custom metrics: categorical accuracy for a vector of predictions
"""def vector_categorical_accuracy(y_true, y_pred):
    #ToDo: implement in Theano
    return K.mean(K.equal([K.argmax(y_true[i, :, :], axis=-1) for i in K.int_shape(y_true)[0]],
                          [K.argmax(y_pred[i, :, :], axis=-1) for i in K.int_shape(y_true)[0]]))"""


def vector_categorical_crossentropy(y_true, y_pred):
    """return categorical_crossentropy(
        K.permute_dimensions(K.batch_flatten(K.permute_dimensions(y_pred, (2,0,1))), (1,0)),
        K.permute_dimensions(K.batch_flatten(K.permute_dimensions(y_true, (2,0,1))), (1,0)))"""
    return categorical_crossentropy(K.reshape(y_true, shape=(-1, 8)),
                                    K.reshape(y_pred, shape=(-1, 8)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process html patents')
    parser.add_argument('--size', '-s', type=str, default='full',
                        help='dataset size')
    parser.add_argument('--num_categories', '-c', type=int, default=8,
                        help='how many categories (3/8)?')
    parser.add_argument('--num_epochs', '-e', type=int, default=10,
                        help='how many epochs?')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01,
                        help='learning rate?')
    parser.add_argument('--batch_size', '-b', type=int, default=128,
                        help='batch size?')

    args = parser.parse_args()

    if args.size == 'full':
        data_id = 'SecS'
    elif args.size == 'small':
        data_id = 'smallSecS'
    else:
        print('Unknown data size: should be full or small')

    SecStructure(args)

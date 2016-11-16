from keras.layers import Input, LSTM, Activation,Convolution1D, MaxPooling1D, \
    Dense, UpSampling1D, Flatten, Reshape, Dropout
from keras.models import Model
from keras.optimizers import SGD, Nadam
from keras.regularizers import l2, activity_l1
from keras.utils.visualize_util import plot
import numpy as np

# Check if package is installed, else fallback to developer mode imports
try:
    from evolutron.tools import load_dataset
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.abspath('..'))

    from evolutron.tools import load_dataset

data_id = 'SecS'
num_categories = 8

dataset, secondary_structure = load_dataset(data_id, padded=True)

num_examples = dataset.shape[0]
example_len = dataset.shape[1]
num_channels = dataset.shape[2]

print(dataset.shape)

input_aa_seq = Input(shape=(example_len, num_channels))

lstm = LSTM(output_dim=num_categories, input_shape=(None, example_len, num_channels),
            return_sequences=True, W_regularizer=l2(0.1))(input_aa_seq)

flat = Flatten()(lstm)

dropout = Dropout(p=0.3)(flat)

dense = Dense(output_dim=example_len*num_categories,
              activity_regularizer=activity_l1(0.1))(dropout)

reshape = Reshape(target_shape=(example_len, num_categories))(dense)

output = Activation(activation='softmax')(reshape)

model = Model(input=input_aa_seq, output=output)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

model.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.summary()

model.fit(dataset, secondary_structure,
          batch_size=128,
          nb_epoch=10,
          shuffle=True,
          validation_split=.2)

model.save('tmp')

plot(model, to_file='model.png')

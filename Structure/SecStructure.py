from keras.layers import Input, LSTM, Activation,Convolution1D, MaxPooling1D, \
    Dense, UpSampling1D, Flatten, Reshape
from keras.models import Model
from keras.optimizers import SGD
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

dataset, secondary_structure = load_dataset(data_id, padded=True)

num_examples = dataset.shape[0]
example_len = dataset.shape[1]
num_channels = dataset.shape[2]
# Check what ordering is keras using
#dataset = dataset.transpose((0, 2, 1))

print(dataset.shape)

# ToDo: load real labels
# load labels
"""secondary_structure = np.zeros(shape=(num_examples, example_len, 3))
for i in range(num_examples):
    tmp = np.random.randint(1, 3, example_len)
    secondary_structure[i, np.arange(example_len), tmp] = 1

print(secondary_structure.shape)"""

# Padding sequences
input_aa_seq = Input(shape=(example_len, num_channels))

lstm = LSTM(3, input_shape=(None, example_len, num_channels),
            return_sequences=True)(input_aa_seq)

flat = Flatten()(lstm)

dense = Dense(example_len*3)(flat)

reshape = Reshape(target_shape=(example_len, 3))(dense)

output = Activation(activation='softmax')(reshape)

model = Model(input=input_aa_seq, output=output)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss='categorical_crossentropy')

model.fit(dataset, secondary_structure,
          batch_size=128,
          nb_epoch=10,
          shuffle=True,
          validation_split=.2)

model.save('tmp')

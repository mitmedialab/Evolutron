Evolutron
=========
A mini deep learning framework for protein evolution.

## Dependencies
* [Theano](https://github.com/Theano/Theano)
* [Lasagne](https://github.com/Lasagne/Lasagne)
* Scikit-Learn


## Packages using Evolutron

* CoMET
* CoHST
* CoBind
* Visualizations


## Test run



```python
from __future__ import print_function, division, absolute_import

import argparse
import time

import numpy as np
seed = 7
np.random.seed(seed)

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
os.chdir('../CoMET')

import evolutron.networks as nets
from evolutron.motifs import motif_extraction
from evolutron.tools import load_dataset, none2str, Handle, shape
from evolutron.engine import DeepTrainer
```

    Using Theano backend.
    Using gpu device 1: GeForce GTX 1080 (CNMeM is enabled with initial size: 80.0% of memory, cuDNN 5105)



```python
# Data and model
data_id = 'zinc'
padded = True
mode = 'unsuper'
model = None

# Architecture
n_filters = 100
filter_length = 30
n_convs = 1
n_fc = 1

# Training
num_epochs = 20
batch_size = 50
optimizer = 'nadam'
rate = .1
validate = .2
```


```python
dataset = load_dataset(data_id, padded=padded)
```

    Dataset size: 11906



```python
x_data = dataset
# Find input shape
if type(x_data) == np.ndarray:
    input_shape = x_data[0].shape
elif type(x_data) == list:
    input_shape = (None, x_data[0].shape[1])
else:
    raise TypeError('Something went wrong with the dataset type')
```


```python
print('Building model ...')
net_arch = nets.DeepCoDER.from_options(input_shape, n_conv_layers=n_convs, n_fc_layers=n_fc,
                                       n_filters=n_filters, filter_length=filter_length)
conv_net = DeepTrainer(net_arch)
conv_net.compile(optimizer=optimizer, lr=rate)
```

    Building model ...



```python
conv_net.display_network_info()
```

    Neural Network has 70320 trainable parameters
      #  Name            Shape       Parameters
    ---  --------------  --------  ------------
      0  aa_seq          7182x20              0
      1  masking_1       7182x20              0
      2  Conv1           7182x100         60100
      3  maxpooling1d_1  1x100                0
      4  flatten_1       100                  0
      5  FCEnc1          100              10100
      6  FCDec1          100                100
      7  reshape_1       1x100                0
      8  Unpooling       7182x100             0
      9  Deconv1         7182x20             20



```python
print('Started training at {}'.format(time.asctime()))
conv_net.fit(x_data[:int(.9 * len(x_data))], x_data[:int(.9 * len(x_data))],
             nb_epoch=num_epochs,
             batch_size=batch_size,
             validate=validate)
```

    Started training at Fri Nov 11 19:52:46 2016
    Train on 8572 samples, validate on 2143 samples
    Epoch 1/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.1825 - mean_cat_acc: 0.9251Epoch 00000: val_loss improved from inf to 0.14314, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 58s - loss: 0.1825 - mean_cat_acc: 0.9250 - val_loss: 0.1431 - val_mean_cat_acc: 0.9286
    Epoch 2/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.1203 - mean_cat_acc: 0.9279Epoch 00001: val_loss improved from 0.14314 to 0.10525, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.1202 - mean_cat_acc: 0.9279 - val_loss: 0.1052 - val_mean_cat_acc: 0.9303
    Epoch 3/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0923 - mean_cat_acc: 0.9291Epoch 00002: val_loss improved from 0.10525 to 0.08421, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.0922 - mean_cat_acc: 0.9292 - val_loss: 0.0842 - val_mean_cat_acc: 0.9310
    Epoch 4/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0779 - mean_cat_acc: 0.9302Epoch 00003: val_loss improved from 0.08421 to 0.07170, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.0779 - mean_cat_acc: 0.9302 - val_loss: 0.0717 - val_mean_cat_acc: 0.9333
    Epoch 5/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0689 - mean_cat_acc: 0.9319Epoch 00004: val_loss improved from 0.07170 to 0.06489, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.0689 - mean_cat_acc: 0.9320 - val_loss: 0.0649 - val_mean_cat_acc: 0.9346
    Epoch 6/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0632 - mean_cat_acc: 0.9326Epoch 00005: val_loss improved from 0.06489 to 0.05998, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.0632 - mean_cat_acc: 0.9326 - val_loss: 0.0600 - val_mean_cat_acc: 0.9351
    Epoch 7/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0580 - mean_cat_acc: 0.9334Epoch 00006: val_loss improved from 0.05998 to 0.05472, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.0580 - mean_cat_acc: 0.9335 - val_loss: 0.0547 - val_mean_cat_acc: 0.9363
    Epoch 8/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0531 - mean_cat_acc: 0.9351Epoch 00007: val_loss improved from 0.05472 to 0.05072, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.0531 - mean_cat_acc: 0.9350 - val_loss: 0.0507 - val_mean_cat_acc: 0.9382
    Epoch 9/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0497 - mean_cat_acc: 0.9369Epoch 00008: val_loss improved from 0.05072 to 0.04806, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.0497 - mean_cat_acc: 0.9370 - val_loss: 0.0481 - val_mean_cat_acc: 0.9398
    Epoch 10/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0476 - mean_cat_acc: 0.9382Epoch 00009: val_loss improved from 0.04806 to 0.04640, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.0476 - mean_cat_acc: 0.9381 - val_loss: 0.0464 - val_mean_cat_acc: 0.9405
    Epoch 11/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0462 - mean_cat_acc: 0.9385Epoch 00010: val_loss improved from 0.04640 to 0.04522, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 54s - loss: 0.0462 - mean_cat_acc: 0.9385 - val_loss: 0.0452 - val_mean_cat_acc: 0.9409
    Epoch 12/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0450 - mean_cat_acc: 0.9392Epoch 00011: val_loss improved from 0.04522 to 0.04419, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.0450 - mean_cat_acc: 0.9391 - val_loss: 0.0442 - val_mean_cat_acc: 0.9414
    Epoch 13/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0442 - mean_cat_acc: 0.9394Epoch 00012: val_loss improved from 0.04419 to 0.04339, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.0442 - mean_cat_acc: 0.9394 - val_loss: 0.0434 - val_mean_cat_acc: 0.9418
    Epoch 14/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0434 - mean_cat_acc: 0.9397Epoch 00013: val_loss improved from 0.04339 to 0.04271, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.0434 - mean_cat_acc: 0.9398 - val_loss: 0.0427 - val_mean_cat_acc: 0.9421
    Epoch 15/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0427 - mean_cat_acc: 0.9403Epoch 00014: val_loss improved from 0.04271 to 0.04199, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.0427 - mean_cat_acc: 0.9403 - val_loss: 0.0420 - val_mean_cat_acc: 0.9427
    Epoch 16/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0420 - mean_cat_acc: 0.9409Epoch 00015: val_loss improved from 0.04199 to 0.04128, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.0420 - mean_cat_acc: 0.9409 - val_loss: 0.0413 - val_mean_cat_acc: 0.9434
    Epoch 17/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0413 - mean_cat_acc: 0.9415Epoch 00016: val_loss improved from 0.04128 to 0.04068, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.0413 - mean_cat_acc: 0.9415 - val_loss: 0.0407 - val_mean_cat_acc: 0.9439
    Epoch 18/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0407 - mean_cat_acc: 0.9421Epoch 00017: val_loss improved from 0.04068 to 0.04010, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.0407 - mean_cat_acc: 0.9420 - val_loss: 0.0401 - val_mean_cat_acc: 0.9445
    Epoch 19/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0402 - mean_cat_acc: 0.9426Epoch 00018: val_loss improved from 0.04010 to 0.03965, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.0402 - mean_cat_acc: 0.9426 - val_loss: 0.0397 - val_mean_cat_acc: 0.9448
    Epoch 20/20
    8550/8572 [============================>.] - ETA: 0s - loss: 0.0398 - mean_cat_acc: 0.9431Epoch 00019: val_loss improved from 0.03965 to 0.03923, saving model to /tmp/best_0.6266421296103689.h5
    8572/8572 [==============================] - 53s - loss: 0.0398 - mean_cat_acc: 0.9431 - val_loss: 0.0392 - val_mean_cat_acc: 0.9458
    Model trained for 20 epochs. Total time: 1078.762s



```python
score = conv_net.score(x_data[int(.9 * len(x_data)):], x_data[int(.9 * len(x_data)):])
print('Test Loss:{0:.6f}, Test Accuracy: {1:.2f}%'.format(score[0], 100*score[1]))
```

    1191/1191 [==============================] - 3s
    Test Loss:0.039633, Test Accuracy: 94.13%



```python
handle = Handle(batch_size=batch_size,
        filter_size=filter_length,
        filters=n_filters,
        program='CoMET',
        dataset=data_id,
        extra=None)
handle.model = 'realDeepCoDER'
conv_net.save_train_history(handle)
conv_net.save_model_to_file(handle)
```

    Model saved in:models/zinc/100_30_20_50_realDeepCoDER.model


# coding=utf-8

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def standard(patience=10, reduce_factor=0.5, min_lr=0.001, monitor='val_loss'):
    es = EarlyStopping(monitor=monitor,
                       min_delta=0.0001,
                       patience=patience,
                       verbose=1,
                       mode='min')
    reduce_lr = ReduceLROnPlateau(monitor=monitor,
                                  factor=reduce_factor,
                                  patience=5,
                                  min_lr=min_lr,
                                  verbose=1)

    callbacks = [es, reduce_lr]

    return callbacks

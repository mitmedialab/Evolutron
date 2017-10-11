import numpy as np
from typing import Generator

from keras.callbacks import Callback
from keras.utils import GeneratorEnqueuer
from sklearn.metrics import precision_score, roc_auc_score


# TODO: Add extra metrics to history so that they are saved in the file
class PrecisionCallback(Callback):
    def on_train_begin(self, logs=None):
        self.prfs = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.validation_data[0])
        self.prfs.append(
                precision_score(np.argmax(self.validation_data[1], axis=1), np.argmax(y_pred, axis=1),
                                average='weighted'))
        print('Precision Score is %s' % self.prfs[-1])
        print('random')


class AucCallback(Callback):
    def on_train_begin(self, logs=None):
        self.auc = []

    def on_epoch_end(self, epoch, logs=None):
        if self.model.generator is None:
            y_pred = self.model.predict(self.validation_data[0])
        else:
            if isinstance(self.validation_data, Generator):
                y_pred = []
                y_true = []

                enqueuer = GeneratorEnqueuer(self.validation_data,
                                             use_multiprocessing=False,
                                             wait_time=.01)
                enqueuer.start(workers=1, max_queue_size=10)
                output_generator = enqueuer.get()

                for _ in range(self.validation_steps):
                    generator_output = next(output_generator)
                    if not hasattr(generator_output, '__len__'):
                        raise ValueError('Output of generator should be a tuple '
                                         '(x, y, sample_weight) '
                                         'or (x, y). Found: ' +
                                         str(generator_output))
                    if len(generator_output) == 2:
                        x, y = generator_output
                    elif len(generator_output) == 3:
                        x, y, _ = generator_output
                    else:
                        raise ValueError('Output of generator should be a tuple '
                                         '(x, y, sample_weight) '
                                         'or (x, y). Found: ' +
                                         str(generator_output))
                    outs = self.model.predict_on_batch(x)

                    y_pred += outs.tolist()
                    y_true += y.tolist()

                enqueuer.stop()
            else:
                # validation_data = self.validation_data[0]

                # if len(self.model.inputs) == 1:
                #     nb_samples = len(self.validation_data[0])
                # else:
                #     nb_samples = len(self.validation_data[0][0])
                # steps = np.ceil(nb_samples / 50)

                y_pred = self.model.predict(self.validation_data[0])
                y_true = self.validation_data[1].astype(np.bool)

        roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        self.auc.append(roc_auc)
        print('AUC Score is %s' % self.auc[-1])
        print('random')

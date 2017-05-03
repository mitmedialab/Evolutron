import numpy as np

from keras.callbacks import Callback
from sklearn.metrics import precision_score, roc_auc_score


class PrecisionCallback(Callback):

    # def __init__(self, validation_data):
    #     # self.validation_data = validation_data
    #     super(ClassificationMetrics, self).__init__()

    def on_train_begin(self, logs=None):
        self.prfs = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.validation_data[0])
        self.prfs.append(precision_score(np.argmax(self.validation_data[1], axis=1), np.argmax(y_pred,axis=1), average='weighted'))
        print('Precision Score is %s' % self.prfs[-1])
        print('random')


class AucCallback(Callback):

    def on_train_begin(self, logs=None):
        self.prfs = []

    def on_epoch_end(self, epoch, logs=None):
        if self.model.generator is None:
            y_pred = self.model.predict(self.validation_data[0])
        else:
            if len(self.model.inputs) == 1:
                nb_samples = len(self.validation_data[0])
            else:
                nb_samples = len(self.validation_data[0][0])

            y_pred = self.model.predict_generator(self.model.generator(self.validation_data[0], batch_size=50),
                                                  steps=np.ceil(nb_samples / 50))

        self.prfs.append(roc_auc_score(y_true=self.validation_data[1].astype(np.bool), y_score=y_pred))
        print('AUC Score is %s' % self.prfs[-1])
        print('random')
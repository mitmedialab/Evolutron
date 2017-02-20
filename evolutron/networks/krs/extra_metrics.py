from keras.metrics import categorical_accuracy, recall, precision, fmeasure
import keras.backend as K
import numpy as np

Beta = 1


def mean_cat_acc(y_true, y_pred):
    nb_categories = K.shape(y_true)[-1]
    y_true = K.reshape(y_true, shape=(-1, nb_categories))
    y_pred = K.reshape(y_pred, shape=(-1, nb_categories))

    real_len = K.sum(y_true)
    is_real = K.sum(y_true, -1)
    #y_true = K.concatenate([y_true, K.epsilon() * K.max(K.ones_like(y_true), -1, keepdims=True)])
    s = K.sum(K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), 'float32') * is_real)
    return s/real_len


def multiclass_acc(y_true, y_pred):
    y_pred = K.round(y_pred)

    return K.sum(K.equal(y_true, y_pred))/K.prod(K.shape(y_true))


def multiclass_precision(y_true, y_pred):
    """This metric return a precision score for each class"""
    nb_categories = K.shape(y_true)[-1]
    precision_dict = {}

    pred_max = K.argmax(y_pred, -1)
    true_max = K.argmax(y_true, -1)

    for i in range(8):
        ten = i * K.equal(pred_max, pred_max)

        true_positives = K.sum(K.equal(pred_max, ten) * K.equal(true_max, ten))
        predicted_positives = K.sum(K.equal(pred_max, ten))
        precision_dict['p_%d' % i] = true_positives / (predicted_positives + K.epsilon())

    return precision_dict


def macro_precision(y_true, y_pred):
    """This metric return a macro-averaged precision score"""
    precision_dict = multiclass_precision(y_true, y_pred)
    nb_categories = len(precision_dict)
    sum = 0

    for v in precision_dict.values():
        sum += v

    return sum / nb_categories


def micro_precision(y_true, y_pred):
    """This metric return a micro-averaged precision score"""
    true_positives = 0
    predicted_positives = 0

    pred_max = K.argmax(y_pred, -1)
    true_max = K.argmax(y_true, -1)

    for i in range(8):
        ten = i * K.equal(pred_max, pred_max)

        true_positives += K.sum(K.equal(pred_max, ten) * K.equal(true_max, ten))
        predicted_positives += K.sum(K.equal(pred_max, ten))

    return true_positives / (predicted_positives + K.epsilon())


def multiclass_recall(y_true, y_pred):
    """This metric return a recall score for each class"""
    nb_categories = K.shape(y_true)[-1]
    recall_dict = {}

    pred_max = K.argmax(y_pred, -1)
    true_max = K.argmax(y_true, -1)

    for i in range(8):
        ten = i * K.equal(pred_max, pred_max)

        true_positives = K.sum(K.equal(pred_max, ten) * K.equal(true_max, ten))
        true_elements = K.sum(K.equal(true_max, ten))
        recall_dict['r_%d' % i] = true_positives / (true_elements + K.epsilon())

    return recall_dict


def macro_recall(y_true, y_pred):
    """This metric return a macro-averaged recall score"""
    recall_dict = multiclass_recall(y_true, y_pred)
    nb_categories = len(recall_dict)
    sum = 0

    for v in recall_dict.values():
        sum += v

    return sum / nb_categories


def micro_recall(y_true, y_pred):
    """This metric return a micro-averaged recall score"""
    true_positives = 0
    true_elements = 0

    pred_max = K.argmax(y_pred, -1)
    true_max = K.argmax(y_true, -1)

    for i in range(8):
        ten = i * K.equal(pred_max, pred_max)

        true_positives += K.sum(K.equal(pred_max, ten) * K.equal(true_max, ten))
        true_elements += K.sum(K.equal(true_max, ten))
    return true_positives / (true_elements + K.epsilon())


def multiclass_fmeasure(y_true, y_pred):
    """This metric return a F-score for each class"""
    fmeasure_dict = {}

    pred_max = K.argmax(y_pred, -1)
    true_max = K.argmax(y_true, -1)

    for i in range(8):
        ten = i * K.equal(pred_max, pred_max)

        true_positives = K.sum(K.equal(pred_max, ten) * K.equal(true_max, ten))
        predicted_positives = K.sum(K.equal(pred_max, ten))
        true_elements = K.sum(K.equal(true_max, ten))

        fmeasure_dict['f_%d' % i] = ((Beta**2 + 1) * true_positives) / (Beta**2 * true_elements + predicted_positives)

    return fmeasure_dict


def macro_fmeasure(y_true, y_pred):
    """This metric return a macro-averaged F-score"""
    precision_value = macro_precision(y_true, y_pred)
    recall_value = macro_recall(y_true, y_pred)

    return ((Beta**2 + 1) * precision_value * recall_value) / (Beta**2 * precision_value + recall_value)


def micro_fmeasure(y_true, y_pred):
    """This metric return a micro-averaged F-score"""
    precision_value = micro_precision(y_true, y_pred)
    recall_value = micro_recall(y_true, y_pred)

    return ((Beta**2 + 1) * precision_value * recall_value) / (Beta**2 * precision_value + recall_value)
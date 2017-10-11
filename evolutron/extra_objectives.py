# coding=utf-8
import keras.backend as K
from keras.objectives import mean_squared_error

Beta = 1


def masked_mse(inp, decoded, mask_value=0.0):
    boolean_mask = K.any(K.not_equal(inp, mask_value), axis=-1, keepdims=True)
    decoded = decoded * K.cast(boolean_mask, K.floatx())
    return mean_squared_error(y_true=inp, y_pred=decoded)


def multiclass_categorical_crossentropy(output, target):
    """
        Categorical crossentropy between an output tensor
        and a target tensor, where the target is a tensor of the same
        shape as the output. Supports multiclass classification.
    """

    # manual computation of crossentropy
    epsilon = K.epsilon()
    output = K.clip(output, epsilon, 1. - epsilon)
    return K.mean(- K.sum(target * K.log(output), axis=-1))


def pred_pos_approx(y_pred, category):
    # nb_categories = K.shape(y_pred)[-1]
    nb_categories = 8

    pp = 1

    for i in range(nb_categories):
        if i != category:
            pp = pp * K.maximum((y_pred[:, category] - y_pred[:, i]), 0)

    return pp


def multiclass_loss_precision(y_true, y_pred):
    """This metric returns a smooth approximation of the precision score for each class"""
    # nb_examples = K.shape(y_true)[0]
    # nb_categories = K.shape(y_true)[-1]
    # nb_examples = K.count_params(y_true) / 8
    nb_categories = 8
    precision_dict = {}

    y_true = K.reshape(y_true, shape=(-1, nb_categories))
    y_pred = K.reshape(y_pred, shape=(-1, nb_categories))

    for i in range(nb_categories):
        true_positives = 0
        predicted_positives = 0

        pp = pred_pos_approx(y_pred, i)
        true_positives += K.sum(y_true[:, i] * pp)
        predicted_positives += K.sum(pp)

        precision_dict['p_%d' % i] = (true_positives + K.epsilon()) / (predicted_positives + K.epsilon())

    return precision_dict


def precision_loss(y_true, y_pred):
    """This function return a macro-averaged precision loss"""
    precision_dict = multiclass_loss_precision(y_true, y_pred)
    nb_categories = len(precision_dict)
    sum = 0

    for v in precision_dict.values():
        sum += v

    return sum / nb_categories


def multiclass_loss_recall(y_true, y_pred):
    """This function return a smooth approximation of the recall score for each class"""
    # nb_examples = K.shape(y_true)[0]
    # nb_categories = K.shape(y_true)[-1]
    # nb_examples = K.sum(y_true)
    nb_categories = 8
    recall_dict = {}

    y_true = K.reshape(y_true, shape=(-1, nb_categories))
    y_pred = K.reshape(y_pred, shape=(-1, nb_categories))

    for i in range(nb_categories):
        true_positives = 0
        true_elements = 0

        pp = pred_pos_approx(y_pred, i)
        true_positives += K.sum(y_true[:, i] * pp)
        true_elements += K.sum(y_true[:, i])

        recall_dict['p_%d' % i] = (true_positives + K.epsilon()) / (true_elements + K.epsilon())

    return recall_dict


def recall_loss(y_true, y_pred):
    """This function return a macro-averaged recall loss"""
    recall_dict = multiclass_loss_precision(y_true, y_pred)
    nb_categories = len(recall_dict)
    sum = 0

    for v in recall_dict.values():
        sum += v

    return sum / nb_categories


def fmeasure_loss(y_true, y_pred):
    """This function return a macro-averaged F-score"""
    precision_value = precision_loss(y_true, y_pred)
    recall_value = recall_loss(y_true, y_pred)

    return ((Beta ** 2 + 1) * precision_value * recall_value) / (Beta ** 2 * precision_value + recall_value)


def focal_loss(alpha=.5, gamma=1):
    return lambda y_true, y_pred: K.mean(-1*alpha*y_true*K.pow(1-y_pred, gamma)*K.log(y_pred) -
                                         (1-alpha)*(1-y_true)*K.pow(y_pred, gamma)*K.log(1-y_pred), axis=-1)

#!/bin/env python
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.metrics import auc

filename = 'networks/m6a/201_1_3_15.k_fold.npz'

with np.load(filename) as f:
    val_losses = f['val_loss']
    train_losses = f['train_loss']
    y_classes = f['y_class']
    y_train_classes = f['y_train_class']
    # val_classes = f['val_class']
    val_preds = f['val_pred']
    train_preds = f['train_pred']
    val_classes = [np.argmax(np.asarray(pred).squeeze(), axis=1) for pred in val_preds]
    n_classes = len(np.unique(y_classes[0]))
    n_folds = len(y_classes)


# noinspection PyShadowingNames
def plot_cm_all_folds(y_classes, val_classes, n_folds=10):
    labels = ['control', 'binders', 'non-binders']
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.reshape(n_folds)
    for i in xrange(n_folds):
        cm = confusion_matrix(y_classes[i], val_classes[i])
        ax[i].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(3)
        ax[i].set_xticks(tick_marks)
        ax[i].set_yticks(tick_marks)
        ax[i].set_xticklabels(labels, rotation=45)
        ax[i].set_yticklabels(labels)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.suptitle('Confusion matrices for each round of cross-validation')


# noinspection PyShadowingNames
def plot_roc_all_folds(y_classes, val_preds, n_folds=10):
    fig2, ax2 = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(15, 15), facecolor='white')
    fig2.subplots_adjust(wspace=0.2, hspace=0.2)
    ax2 = ax2.reshape(n_folds)
    for fold in xrange(n_folds):
        y = y_classes[fold]
        y = np.asarray(y)
        pred = val_preds[fold]
        pred = np.squeeze(pred)
        for i in xrange(n_classes):
            fpr, tpr, _ = roc_curve(y, pred[:, i], pos_label=i)
            ax2[fold].plot(fpr, tpr, label='ROC of class {0}'.format(i))
        ax2[fold].plot([0, 1], [0, 1], 'k--')
        ax2[fold].set_xlim([0.0, 1.0])
        ax2[fold].set_ylim([0.0, 1.0])
        ax2[fold].set_xlabel('False Positive Rate')
        ax2[fold].set_ylabel('True Positive Rate')
        ax2[fold].set_title('Fold {0}'.format(fold + 1))
    ax2[0].legend(bbox_to_anchor=(0., 1.02, 1., .3), loc=3, borderaxespad=0., mode='expand', ncol=3)
    fig2.suptitle('ROC curves')


#  Micro-average Precision-recall curve and AUC
y = np.concatenate([y for y in y_classes])
pred = np.concatenate([p for p in val_preds])
pred = np.squeeze(pred)
plt.figure(3)
for i in xrange(n_classes):
    prec, recall, _ = precision_recall_curve(y, pred[:, i], pos_label=i)
    auc_prc = auc(recall, prec)
    plt.plot(recall, prec, label='Precision-recall curve of class {0} AUC={1:0.2f}'.format(i, auc_prc))
plt.legend(loc="lower right")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve'.format())
plt.show()

#  Micro-average Precision-recall curve and AUC
y_train = np.concatenate([y for y in y_train_classes])
train_pred = np.concatenate([p for p in train_preds])
train_pred = np.squeeze(train_pred)
plt.figure(4)
for i in xrange(n_classes):
    prec, recall, _ = precision_recall_curve(y_train, train_pred[:, i], pos_label=i)
    auc_prc = auc(recall, prec)
    plt.plot(recall, prec, label='Precision-recall curve of class {0} AUC={1:0.2f}'.format(i, auc_prc))
plt.legend(loc="lower right")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve'.format())
plt.show()

# Micro-average ROC curve and ROC area
y = np.concatenate([y for y in y_classes])
pred = np.concatenate([p for p in val_preds])
pred = np.squeeze(pred)
plt.figure(5)
for i in xrange(n_classes):
    fpr, tpr, _ = roc_curve(y, pred[:, i], pos_label=i)
    auc_roc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve of class {0} AUC={1:0.2f}'.format(i, auc_roc))
plt.plot([0, 1], [0, 1], 'k--')
plt.legend(loc="lower right")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive')
plt.ylabel('True positive')
plt.title('ROC curve')
plt.show()

#  Micro-average Precision-recall curve and AUC
y_train = np.concatenate([y for y in y_train_classes])
train_pred = np.concatenate([p for p in train_preds])
train_pred = np.squeeze(train_pred)
plt.figure(6)
for i in xrange(n_classes):
    fpr, tpr, _ = roc_curve(y_train, train_pred[:, i], pos_label=i)
    auc_roc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve of class {0} AUC={1:0.2f}'.format(i, auc_roc))
plt.plot([0, 1], [0, 1], 'k--')
plt.legend(loc="lower right")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive')
plt.ylabel('True positive')
plt.title('ROC curve')
plt.show()

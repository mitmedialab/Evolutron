# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_history(model):
    if isinstance(model, str):
        with np.load(model) as f:
            train_err_mem = f['train_err_mem']
            valid_err_mem = f['val_err_mem']
    else:
        train_err_mem = model.train_err_mem
        valid_err_mem = model.val_err_mem

    plt.figure(3)
    plt.plot(train_err_mem, label='Train loss')
    plt.plot(valid_err_mem, label='Valid loss')
    plt.legend(loc="best")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training history')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def plot_training_history(filename):

    f = np.load(filename)

    ep = len(f['train_err_mem'])

    plt.figure(3)
    plt.plot(range(1,ep+1), f['train_err_mem'], label='Train set')
    plt.plot(range(1,ep+1), f['val_err_mem'], label='Valid set')
    plt.legend(loc="upper right")

    plt.xlabel('Epochs')
    plt.ylabel('Cost Funtion')
    plt.title('Training history')
    plt.show()


filename = 'models/b1h/150_64_15.history.npz'
plot_training_history(filename)
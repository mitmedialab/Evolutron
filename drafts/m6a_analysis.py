import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr

from evolutron.tools import load_dataset
from evolutron.tools.io_tools import m6a

raw_data = m6a(padded=False, probe='1')
x_train, y_train = load_dataset(raw_data, shuffled=True)

x_len = np.asarray([x.shape[1] for x in x_train])

y_arr_1 = [y[0] for y in y_train]

print(pearsonr(y_arr_1, np.log(x_len)))

fig = plt.figure()

ax1 = fig.add_subplot(211)
n, bins, patches = ax1.hist(x_len, 50, normed=1, facecolor='green', alpha=0.75)

ax2 = fig.add_subplot(212)
ax2.scatter(y_arr_1, np.log(x_len))

plt.tight_layout()
fig = plt.gcf()

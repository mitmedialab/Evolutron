# coding=utf-8

import os

from evolutron.tools import Handle

# x = os.listdir('networks/CoMET/')
# x.remove('c2h2')
# x.remove('cohst')
#
# for t in x:
fold = 'networks/CoMET/type2p/'
for f in os.listdir(fold):

    h = Handle.from_filename(fold + f)

    print(fold + h.filename + '.npz')
    print(fold + f)
    # os.rename(fold + f,fold + h.filename + '.npz' )
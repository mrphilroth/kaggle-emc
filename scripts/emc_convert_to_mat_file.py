#!/usr/bin/python
#

import EMC_IO
import numpy as np
from scipy.io import savemat

def row_normalize(d) :
    i = 0
    srow = 0
    for erow in d.indptr[1:] :
        if srow != erow : d.data[srow:erow] = d.data[srow:erow] / np.sum(d.data[srow:erow])
        srow = erow
        i += 1
    return d

ddir = "/home/proth/kaggle/EMCsourcecode/data/"
d = EMC_IO.EMC_ReadData(ddir + "train_data.csv")
savemat(ddir + "train_data.mat", {'a': d})
drn = row_normalize(d)
savemat(ddir + "train_data_row_normalized.mat", {'a': drn})

dtest = EMC_IO.EMC_ReadData(ddir + "test_data.csv")
savemat(ddir + "test_data.mat", {'a': dtest})
dtestrn = row_normalize(dtest)
savemat(ddir + "test_data_row_normalized.mat", {'a': dtestrn})

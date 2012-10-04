#!/usr/bin/python
#

import EMC_IO
import utility
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

d = EMC_IO.EMC_ReadData(utility.ddir + "/train_data.csv")
savemat(utility.ddir + "/train_data.mat", {'a': d})
drn = row_normalize(d)
savemat(utility.ddir + "/train_data_row_normalized.mat", {'a': drn})

dtest = EMC_IO.EMC_ReadData(utility.ddir + "/test_data.csv")
savemat(utility.ddir + "/test_data.mat", {'a': dtest})
dtestrn = row_normalize(dtest)
savemat(utility.ddir + "/test_data_row_normalized.mat", {'a': dtestrn})

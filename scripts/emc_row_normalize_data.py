#!/usr/bin/python
#

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

d = utility.get_train_data(rn=False, csr=True)
d = row_normalize(d)
savemat(utility.ddir + "/train_data_row_normalized.mat", {'a': d})

t = utility.get_test_data(rn=False, csr=True)
t = row_normalize(t)
savemat(utility.ddir + "/test_data_row_normalized.mat", {'a': t})

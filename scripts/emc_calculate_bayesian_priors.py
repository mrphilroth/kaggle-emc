#!/usr/bin/python
#

import utility
import numpy as np
from scipy.io import savemat

d = utility.get_train_data(rn=False, csr=True)
l = utility.get_train_labels()
a = np.arange(len(l))

nc = np.max(l) + 1
nw = d.shape[1]
bp = np.reshape(np.zeros(nc * nw), (nc, nw))

for ic in range(nc) :
    dc = d[a[l == ic]]
    s = np.array(dc.sum(axis=0))[0]
    bp[ic] = (1.0 + s) / (np.sum(s) + nw)

savemat(utility.ddir + "/train_data_bayesian_priors.mat", {'a': bp})

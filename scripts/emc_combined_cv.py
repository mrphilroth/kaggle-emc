#!/usr/bin/python
#

import utility
import numpy as np
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Get Data
global d, l, nwords, bprobs, rdist, nc, ncv, cvsets
d = utility.get_train_data(rn=False, csr=True)
l = utility.get_train_labels()
nwords = np.array(d.sum(axis = 1))
bprobs = utility.get_bayesian_probs()
rdist = utility.get_centroid_dist(dset="train")
nc = np.max(l) + 1
ncv = 5
cvsets = utility.def_cross_validate_sets(d.shape[0], ncv)

# Cross Validation
def do_cv_set(icv) :

    # New Sets
    indtest = cvsets[icv]
    indtrain = np.array([], dtype=np.int32)
    for i in range(ncv) :
        if i != icv : indtrain = np.hstack((indtrain, cvsets[i]))
    dtrain = np.hstack((bprobs[indtrain], rdist[indtrain]))
    dtest = np.hstack((bprobs[indtest], rdist[indtest]))

    ltest = l[indtest]
    ltrain = l[indtrain]
    atrain = np.arange(dtrain.shape[0])
    ntest = dtest.shape[0]

    # Linear SVC
    lsvc = LinearSVC()
    lsvc.fit(dtrain, ltrain)

    # Predict
    sub = lsvc.predict_proba(dtest)
    
    # Clean for submitting
    sub = utility.clean_submission(sub)
    utility.write_cv_submission("combined", icv, sub, ltest)

    # Evaluate submission
    return icv, utility.logloss(sub, ltest)

# Multiprocessing
logloss = np.zeros(ncv)
pool = multiprocessing.Pool(ncv)
for icv, ll in pool.imap_unordered(do_cv_set, np.arange(ncv), chunksize=1) :
    logloss[icv] = ll
print logloss

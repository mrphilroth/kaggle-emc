#!/usr/bin/python
#

import utility
import numpy as np
import multiprocessing
from sklearn.linear_model import LogisticRegression

# Get Data
d = utility.get_train_data(rn=False, csr=True)
l = utility.get_train_labels()
a = np.arange(len(l))
nc = np.max(l) + 1
cs = np.arange(nc)
nw = d.shape[1]
ncv = 5
cvsets = utility.def_cross_validate_sets(d.shape[0], ncv)

# Bayesian Probability
def get_bayesian_prob(d, bplog, cps) :
    i = 0
    srow = 0
    bprobs = np.reshape(np.zeros(d.shape[0] * len(cps)), (d.shape[0], len(cps)))
    for erow in d.indptr[1:] :
        ibprobs = np.zeros(nc, dtype=np.float64)
        iws = d.indices[srow:erow]
        nws = d.data[srow:erow]
        for ic in range(nc) :
            probs = np.sum(nws * bplog[ic][iws])
            # probs = np.sum(bplog[ic][iws])
            ibprobs[ic] = probs + np.log10(cps[ic])

        ibprobs = ibprobs - np.max(ibprobs)
        # ibprobs = -ibprobs / np.min(ibprobs)  # Less certain
        ibprobs = np.power(10.0, ibprobs)
        bprobs[i] = ibprobs
        i += 1
        srow = erow
    return bprobs

# Cross Validation
def do_cv_set(icv) :

    # New Sets
    indtest = cvsets[icv]
    indtrain = np.array([], dtype=np.int32)
    for i in range(ncv) :
        if i != icv : indtrain = np.hstack((indtrain, cvsets[i]))
    dtest = d[indtest]   
    dtrain = d[indtrain]
    ltest = l[indtest]
    ltrain = l[indtrain]
    atrain = np.arange(dtrain.shape[0])
    ntest = dtest.shape[0]

    # Calculate Bayesian Priors
    bp = np.reshape(np.zeros(nc * nw), (nc, nw))
    for ic in range(nc) :
        dc = dtrain[atrain[ltrain == ic]]
        s = np.array(dc.sum(axis=0))[0]
        bp[ic] = (1.0 + s) / (np.sum(s) + nw)
    bplog = np.log10(bp)

    # Use on training set
    cps = np.array([np.sum(ltrain == ic) for ic in cs], dtype=np.float64) / len(ltrain)
    mbtrain = get_bayesian_prob(dtrain, bplog, cps)

    # Input that to a Logistic Regression
    lr = LogisticRegression()
    lr.fit(mbtrain, ltrain)

    # Use on test set
    mbtest = get_bayesian_prob(dtest, bplog, cps)
    lrtest = lr.predict_proba(mbtest)

    # Clean for submitting
    sub = utility.clean_submission(lrtest)

    print "Example prediction"
    print zip(np.arange(sub.shape[1]), sub[0])
    print "actual", ltest[0]
    print "ordered guesses", np.arange(sub.shape[1])[np.argsort(sub[0])][:-5:-1]

    utility.write_cv_submission("regressed_bayesian_test", icv, sub, ltest)
    
    # Evaluate submission
    ll = utility.logloss(sub, ltest)

    return icv, ll

# Multiprocess the Cross Validation
# do_cv_set(0)
logloss = np.zeros(ncv)
pool = multiprocessing.Pool(ncv)
for icv, ll in pool.imap_unordered(do_cv_set, np.arange(ncv), chunksize=1) :
    logloss[icv] = ll

print logloss


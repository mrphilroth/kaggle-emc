#!/usr/bin/python
#

import utility
import numpy as np
import multiprocessing
from scipy.optimize import fmin

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
rank = 9

def func(x, ps, acts) :
    p = np.polynomial.Polynomial(np.hstack([x, [1 - np.sum(x), 0]]))
    s = p(ps)
    s = s / np.sum(s, axis=1).reshape((1, -1)).T
    s = np.clip(s, 1.0e-15, 1.0 - 1.0e-15)
    ll = 0.0
    for i in range(len(acts)) : ll += np.log(s[i][acts[i]])
    return - ll / len(acts)

# Cross Validation
def do_cv_set(icv) :

    # New Sets
    indtest = cvsets[icv]
    indtrain = np.array([], dtype=np.int32)
    for i in range(ncv) :
        if i != icv : indtrain = np.hstack((indtrain, cvsets[i]))
    # dtrain = np.hstack((bprobs[indtrain], rdist[indtrain]))
    # dtest = np.hstack((bprobs[indtest], rdist[indtest]))
    dtrain = bprobs[indtrain]
    dtest = bprobs[indtest]
    ltest = l[indtest]
    ltrain = l[indtrain]
    atrain = np.arange(dtrain.shape[0])
    ntest = dtest.shape[0]

    # Optimize the results
    res = fmin(func, np.zeros(rank), args=(dtrain, ltrain))
    p = np.polynomial.Polynomial(np.hstack([res, [1 - np.sum(res), 0]]))
    sub = utility.clean_submission(p(dtest))
    
    # Submission
    print "Example prediction"
    print zip(np.arange(sub.shape[1]), sub[0])
    print "actual", ltest[0]
    print "ordered guesses", np.arange(sub.shape[1])[np.argsort(sub[0])][:-5:-1]
    print res

    utility.write_cv_submission("optimized_regressed_bayesian", icv, sub, ltest)
    
    # Evaluate submission
    ll = utility.logloss(sub, ltest)
    print ll
    return icv, ll, res

# Multiprocessing
logloss = np.zeros(ncv)
fres = np.reshape(np.zeros(rank * ncv), (ncv, rank))
pool = multiprocessing.Pool(ncv)
for icv, ll, res in pool.imap_unordered(do_cv_set, np.arange(ncv), chunksize=1) :
    logloss[icv] = ll
    fres[icv] = res
print logloss
print fres

# Plot the functions
import pylab as plt
x = np.linspace(0, 1, 100)
plt.plot(x, x, color='k', ls='--')
for icv in range(ncv) :
    p = np.polynomial.Polynomial(np.hstack([fres[icv], [1 - np.sum(fres[icv]), 0]]))
    plt.plot(x, p(x))
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.savefig("optimized_functions.png")

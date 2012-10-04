#!/usr/bin/python
#

import utility
import multiprocessing
import numpy as np
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression

# Get Data
global d, l, nc, ncv, cvsets
d = utility.get_train_data(rn=True, csr=True)
l = utility.get_train_labels()
nc = np.max(l) + 1
ncv = 5
cvsets = utility.def_cross_validate_sets(d.shape[0], ncv)
dist = utility.get_centroid_dist("train")

# Calculate distance to all category centroids
def get_centroid_dist(d, centroids, icv) :
    sub = np.reshape(np.zeros(nc * d.shape[0]), (d.shape[0], nc))
    for i in range(d.shape[0]) :
        if i % 1000 == 0 : print "%i: i %i of %i" % (icv, i, d.shape[0])
        td = d[i]
        std = td.copy()
        for ic in range(nc - 1) : std = sparse.vstack([std, td])
        diff = centroids - std
        diff.data = np.square(diff.data)
        sumsq = diff.sum(axis=1)
        distance = np.sqrt(np.array(np.transpose(sumsq))[0])
        sub[i] = distance
    return sub

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

    disttrain = dist[indtrain]
    disttest = dist[indtest]

    # Feed to logistic regression
    lr = LogisticRegression()
    lr.fit(dtctrain, ltrain)

    # Use on test set
    dtctest = get_centroid_dist(dtest, cent, icv)
    lrtest = lr.predict_proba(dtctest)
    
    # Clean for submitting
    sub = utility.clean_submission(lrtest)
    utility.write_cv_submission("regressed_centroid_distance", icv, sub, ltest)
    
    # Evaluate submission
    return icv, utility.logloss(sub, ltest)

# Multiprocessing
logloss = np.zeros(ncv)
pool = multiprocessing.Pool(ncv)
for icv, ll in pool.imap_unordered(do_cv_set, np.arange(ncv), chunksize=1) :
    logloss[icv] = ll
print logloss

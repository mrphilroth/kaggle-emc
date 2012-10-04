#!/usr/bin/python
#

import utility
import numpy as np
import multiprocessing
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
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

    # Fit the Bayesian Classifier
    mb = MultinomialNB()
    mb.fit(dtrain, ltrain)

    # Predict the training data
    mbtrain = mb.predict_proba(dtrain)

    # Input that to a Logistic Regression
    lr = LogisticRegression()
    lr.fit(mbtrain, ltrain)

    # Use on test set
    mbtest = mb.predict_proba(dtest)
    lrtest = lr.predict_proba(mbtest)

    # Clean for submitting
    sub = utility.clean_submission(lrtest)

    print "Example prediction"
    print zip(np.arange(sub.shape[1]), sub[0])
    print "actual", ltest[0]
    print "ordered guesses", np.arange(sub.shape[1])[np.argsort(sub[0])][:-5:-1]

    utility.write_cv_submission("regressed_bayesian", icv, sub, ltest)
    
    # Evaluate submission
    ll = utility.logloss(sub, ltest)
    print ll
    return icv, ll

# Multiprocessing
logloss = np.zeros(ncv)
pool = multiprocessing.Pool(ncv)
for icv, ill in pool.imap_unordered(do_cv_set, np.arange(ncv), chunksize=1) :
    logloss[icv] = ill
print logloss

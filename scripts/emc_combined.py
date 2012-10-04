#!/usr/bin/python
#

import utility
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Get Data
d = utility.get_train_data(rn=False, csr=True)
nwords = np.array(d.sum(axis = 1))
bprobs = utility.get_bayesian_probs()
rdist = utility.get_centroid_dist(dset="train")
l = utility.get_train_labels()
train = np.hstack((bprobs, rdist, nwords))

# Random Forest
rf = RandomForestClassifier()
rf.fit(train, l)

# Test Data
t = utility.get_test_data()
tnwords = np.array(t.sum(axis = 1))
tbprobs = utility.read_submission("naive_bayesian_with_regression_sklearn.csv")
trdist = utility.get_centroid_dist(dset="train")

# Predict
test = np.hstack((tbprobs, trdist, tnwords))
sub = rf.predict_proba(test)

# Clean and write submission
sub = utility.clean_submission(sub)
utility.write_submission("combined_submission.csv", sub)

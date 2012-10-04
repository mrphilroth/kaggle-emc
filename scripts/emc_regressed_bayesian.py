#!/usr/bin/python
#

import utility
import numpy as np
from scipy.io import savemat
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression

# Get Data
d = utility.get_train_data(rn=False, csr=True)
l = utility.get_train_labels()
dtest = utility.get_test_data(rn=False, csr=True)

# Fit the Bayesian Classifier
mb = MultinomialNB()
mb.fit(d, l)

# Predict the training data
mbtrain = mb.predict_proba(d)
    
# Input that to a Logistic Regression
lr = LogisticRegression()
lr.fit(mbtrain, l)
lrtrain = lr.predict_proba(mbtrain)
lrtrain = utility.clean_submission(lrtrain)

# Save the training output
savemat(utility.ddir + "/regressed_bayesian_probs.mat", {'a': lrtrain})

# Use on test set
mbtest = mb.predict_proba(dtest)
lrtest = lr.predict_proba(mbtest)

# Clean and write submission
sub = utility.clean_submission(lrtest)
utility.write_submission("naive_bayesian_with_regression_sklearn.csv", sub)

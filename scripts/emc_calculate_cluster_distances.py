#!/usr/bin/python
#

import utility
import numpy as np
from scipy import sparse
from scipy.io import savemat
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression

# Get Data
d = utility.get_train_data(rn=True, csr=True)
t = utility.get_test_data(rn=True, csr=True)
l = utility.get_train_labels()
a = np.arange(d.shape[0])
nc = np.max(l) + 1

# Calculate distance to all category centroids
def get_centroid_dist(fd, centroids) :
    sub = np.reshape(np.zeros(nc * fd.shape[0]), (fd.shape[0], nc))
    for i in range(d.shape[0]) :
        if i % 1000 == 0 : print "i %i of %i" % (i, fd.shape[0])
        td = fd[i]
        std = td.copy()
        for ic in range(nc - 1) : std = sparse.vstack([std, td])
        diff = centroids - std
        diff.data = np.square(diff.data)
        sumsq = diff.sum(axis = 1)
        distance = np.sqrt(np.array(np.transpose(sumsq))[0])
        sub[i] = distance
    return sub

# Calculate the Category centroids
centroids = np.reshape(np.zeros(nc * d.shape[1]), (nc, d.shape[1]))
for ic in range(nc) :
    dc = d[a[l == ic]]
    v = np.array(dc.sum(axis = 0))[0] / np.sum(l == ic)
    centroids[ic] = v
cent = sparse.csr_matrix(centroids)

# Get training data distance to centroids and write to file
dist = get_centroid_dist(d, cent)
savemat(utility.ddir + "/centroid_distance_train.mat", {'a': dist})

# Feed to logistic regression
lr = LogisticRegression()
lr.fit(dist, l)
rdist = lr.predict_proba(dist)
savemat(utility.ddir + "/regressed_centroid_distance_train.mat", {'a': rdist})

# Get test data distance to centroids and write to file
tdist = get_centroid_dist(t, cent)
savemat(utility.ddir + "/centroid_distance_test.mat", {'a': tdist})
trdist = lr.predict_proba(tdist)
savemat(utility.ddir + "/regressed_centroid_distance_test.mat", {'a': trdist})

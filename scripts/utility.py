#!/usr/bin/python
#

import os, sys
from os.path import *
import numpy as np
from scipy import sparse
from scipy.io import loadmat, savemat

bdir = dirname(realpath(__file__))
ddir = abspath(bdir + "/../data/")
cvdir = abspath(bdir + "/../cvsubmissions/")
subdir = abspath(bdir + "/../submissions/")

if not exists(ddir) : os.mkdir(ddir)
if not exists(cvdir) : os.mkdir(cvdir)
if not exists(subdir) : os.mkdir(subdir)

trainfn = "train_data.mat"
trainrnfn = "train_data_row_normalized.mat"
def get_train_data(rn=True, csr=True) :
    fn = trainfn
    if rn : fn = trainrnfn
    m_dict = loadmat(join(ddir, fn))
    d = m_dict['a']
    if csr : return d.tocsr()
    else : return d

testfn = "test_data.mat"
testrnfn = "test_data_row_normalized.mat"
def get_test_data(rn=True, csr=True) :
    fn = testfn
    if rn : fn = testrnfn
    m_dict = loadmat(join(ddir, fn))
    d = m_dict['a']
    if csr : return d.tocsr()
    else : return d

def get_centroid_dist(dset="train", regressed=True) :
    if not dset in ["train", "test"] : return None
    fn = "centroid_distance_%s.mat" % dset
    if regressed : fn = "regressed_centroid_distance_%s.mat" % dset
    m_dict = loadmat(join(ddir, fn))
    return m_dict['a']

def get_bayesian_probs() :
    fn = "regressed_bayesian_probs.mat"
    m_dict = loadmat(join(ddir, fn))
    return m_dict['a']

trainbpfn = "train_data_bayesian_priors.mat"
def get_bayesian_priors() :
    m_dict = loadmat(join(ddir, trainbpfn))
    return m_dict['a']

trainlabfn = "train_labels.csv"
def get_train_labels() :
    data = np.loadtxt(open(join(ddir, trainlabfn), "r"), dtype='int', delimiter = ",")
    return data

def def_cross_validate_sets(n = 175315, nsets = 5) :
    return np.reshape(np.random.permutation(n), (nsets, n / nsets))

def clean_submission(s) :
    s = np.array(s, dtype=np.float64)
    s = s / np.sum(s, axis=1).reshape((1, -1)).T
    s = np.clip(s, 1.0e-15, 1.0 - 1.0e-15)
    return np.array(s, dtype=np.float64)

def logloss(s, actual) :
    result = 0.0
    for itest in range(len(actual)) :
        result += np.log(s[itest][actual[itest]])
    result /= len(actual)
    return -result

def write_cv_submission(fn, i, s, actual) :
    nc = s.shape[1]
    ofile = open(join(cvdir, fn + "_sub%i.csv" % i), "w")
    ofile.write('"id"')
    for ic in range(nc) : ofile.write(',"class%i"' % ic)
    ofile.write("\n")
    for itest in range(s.shape[0]) :
        ofile.write("%i" % itest)
        for ic in range(nc) : ofile.write(",%.20f" % s[itest][ic])
        ofile.write("\n")
    ofile.close()

    afile = join(cvdir, fn + "_act%i.csv" % i)
    actual.tofile(afile, sep=",")

def write_submission(fn, s) :
    nc = s.shape[1]
    ofile = open(join(subdir, fn), "w")
    ofile.write('"id"')
    for ic in range(nc) : ofile.write(',"class%i"' % ic)
    ofile.write("\n")
    for itest in range(s.shape[0]) :
        ofile.write("%i" % (itest + 1))
        for ic in range(nc) : ofile.write(",%.20f" % s[itest][ic])
        ofile.write("\n")
    ofile.close()

def read_submission(fn, cv=False) :
    fullfn = ""
    if cv : fullfn = join(cvdir, fn)
    else : fullfn = join(subdir, fn)
    ifile = open(fullfn, "r")
    l = ifile.readline()

    sub = []
    for line in ifile :
        inums = np.fromstring(line, dtype=np.float64, sep=",")
        sub.append(inums[1:])
    return np.array(sub)

def read_cv_submission(fn, i) :
    subfn = fn + "_sub%i.csv" % i
    sub = read_submission(fn, True)

    actfn = join(cvdir, fn + "_act%i.csv" % i)
    actual = np.fromfile(fn)

    return sub, actual 


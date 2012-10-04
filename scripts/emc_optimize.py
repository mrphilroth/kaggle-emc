#!/usr/bin/python
#

import utility
import numpy as np
import multiprocessing
from scipy.optimize import fmin

rank = 9
def func(x, ps, acts) :
    p = np.polynomial.Polynomial(np.hstack([x, [1 - np.sum(x), 0]]))
    s = p(ps)
    s = s / np.sum(s, axis=1).reshape((1, -1)).T
    s = np.clip(s, 1.0e-15, 1.0 - 1.0e-15)
    ll = 0.0
    for i in range(len(acts)) : ll += np.log(s[i][acts[i]])
    return - ll / len(acts)

# Optimize
bprobs = utility.get_bayesian_probs()
l = utility.get_train_labels()
res = fmin(func, np.zeros(rank), args=(bprobs, l))

# Convert the test set
osub = utility.read_submission("naive_bayesian_with_regression_sklearn.csv")
p = np.polynomial.Polynomial(np.hstack([res, [1 - np.sum(res), 0]]))
sub = utility.clean_submission(p(osub))
utility.write_submission("naive_bayesian_with_optimized_regression_sklearn.csv", sub)

# Plot the function
import pylab as plt
x = np.linspace(0, 1, 100)
plt.plot(x, x, color='k', ls='--')
plt.plot(x, p(x))
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.savefig("optimized_function.png")

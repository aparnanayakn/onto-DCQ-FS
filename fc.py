
import sys
import os
import argparse
import numpy as np

def entropy(vec, base=2):
    q, vec = np.unique(vec, return_counts=True)
    prob_vec = np.array(vec/float(sum(vec)))
    if base == 2:
        logfn = np.log2
    elif base == 10:
        logfn = np.log10
    else:
        logfn = np.log
    return prob_vec.dot(-logfn(prob_vec))

def conditional_entropy(x, y):
    "Returns H(X|Y)."
    uy, uyc = np.unique(y, return_counts=True)
    prob_uyc = uyc/float(sum(uyc))
    cond_entropy_x = np.array([entropy(x[y == v]) for v in uy])
    return prob_uyc.dot(cond_entropy_x)

def mutual_information(x, y):
    " Returns the information gain/mutual information [H(X)-H(X|Y)] between two random vars x & y."
    return entropy(x) - conditional_entropy(x, y)

def symmetrical_uncertainty(x, y):
    " Returns 'symmetrical uncertainty' (SU) - a symmetric mutual information measure."
    return 2.0*mutual_information(x, y)/(entropy(x) + entropy(y))

def c_correlation(X, y):
    su = np.zeros(X.shape[1])
    for i in np.arange(X.shape[1]):
        su[i] = symmetrical_uncertainty(X.iloc[:, i], y)
    return su

def fcbf(X, y):
	"""
	Perform Fast Correlation-Based Filter solution (FCBF).
	
	Parameters:
	-----------
	X : 2-D ndarray
		Feature matrix
	y : ndarray
		Class label vector
	thresh : float
		A value in [0,1) used as threshold for selecting 'relevant' features. 
		A negative value suggest the use of minimum SU[i,c] value as threshold.
	
	Returns:
	--------
	sbest : 2-D ndarray
		An array containing SU[i,c] values and feature index i.
	"""
	n = X.shape[1]
	slist = np.zeros((n, 3))
	slist[:, -1] = 1

	# identify relevant features
	slist[:, 0] = c_correlation(X, y)  # compute 'C-correlation'
	idx = slist[:, 0].argsort()[::-1]
	slist = slist[idx, ]
	slist[:, 1] = idx
	return slist[:,0]
	#return sbest

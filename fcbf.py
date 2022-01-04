#!/usr/bin/env python
# coding: utf-8

# In[1]:



import sys
import os
import argparse
import numpy as np


# In[13]:


def entropy(vec, base=2):
    " Returns the empirical entropy H(X) in the input vector."
    _, vec = np.unique(vec, return_counts=True)
    print("Coming here")
    prob_vec = np.array(vec/float(sum(vec)))
    if base == 2:
        logfn = np.log2
    elif base == 10:
        logfn = np.log10
    else:
        logfn = np.log
    return prob_vec.dot(-logfn(prob_vec))


# In[14]:


def conditional_entropy(x, y):
    "Returns H(X|Y)."
    print("Coming here too")
    uy, uyc = np.unique(y, return_counts=True)
    prob_uyc = uyc/float(sum(uyc))
    cond_entropy_x = np.array([entropy(x[y == v]) for v in uy])
    return prob_uyc.dot(cond_entropy_x)


# In[24]:


def mutual_information(x, y):
    " Returns the information gain/mutual information [H(X)-H(X|Y)] between two random vars x & y."
    return entropy(x) - conditional_entropy(x, y)


# In[25]:


def symmetrical_uncertainty(x, y):
    " Returns 'symmetrical uncertainty' (SU) - a symmetric mutual information measure."
    return 2.0*mutual_information(x, y)/(entropy(x) + entropy(y))


# In[26]:


def getFirstElement(d):
	"""
	Returns tuple corresponding to first 'unconsidered' feature
	
	Parameters:
	----------
	d : ndarray
		A 2-d array with SU, original feature index and flag as columns.
	
	Returns:
	-------
	a, b, c : tuple
		a - SU value, b - original feature index, c - index of next 'unconsidered' feature
	"""

	t = np.where(d[:, 2] > 0)[0]
	if len(t):
		return d[t[0], 0], d[t[0], 1], t[0]
	return None, None, None


# In[27]:


def getNextElement(d, idx):
	"""
	Returns tuple corresponding to the next 'unconsidered' feature.
	
	Parameters:
	-----------
	d : ndarray
		A 2-d array with SU, original feature index and flag as columns.
	idx : int
		Represents original index of a feature whose next element is required.
		
	Returns:
	--------
	a, b, c : tuple
		a - SU value, b - original feature index, c - index of next 'unconsidered' feature
	"""
	t = np.where(d[:, 2] > 0)[0]
	t = t[t > idx]
	if len(t):
		return d[t[0], 0], d[t[0], 1], t[0]
	return None, None, None


# In[28]:


def removeElement(d, idx):
	"""
	Returns data with requested feature removed.
	
	Parameters:
	-----------
	d : ndarray
		A 2-d array with SU, original feature index and flag as columns.
	idx : int
		Represents original index of a feature which needs to be removed.
		
	Returns:
	--------
	d : ndarray
		Same as input, except with specific feature removed.
	"""
	d[idx, 2] = 0
	return d


# In[48]:


def c_correlation(X, y):
    su = np.zeros(X.shape[1])
    for i in np.arange(X.shape[1]):
        su[i] = symmetrical_uncertainty(X.iloc[:, i], y)
    return su


# In[51]:


def fcbf(X, y, thresh):
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
	if thresh < 0:
		thresh = np.median(slist[-1, 0])
		print("Using minimum SU value as default threshold: {0}".format(thresh))
	elif thresh >= 1 or thresh > max(slist[:, 0]):
		print("No relevant features selected for given threshold.")
		print("Please lower the threshold and try again.")
		exit()

	slist = slist[slist[:, 0] > thresh, :]  # desc. ordered per SU[i,c]

	# identify redundant features among the relevant ones
	cache = {}
	m = len(slist)
	p_su, p, p_idx = getFirstElement(slist)
	for i in range(m):
		p = int(p)
		q_su, q, q_idx = getNextElement(slist, p_idx)
		if q:
			while q:
				q = int(q)
				if (p, q) in cache:
					pq_su = cache[(p, q)]
				else:
					pq_su = symmetrical_uncertainty(X.iloc[:, p], X.iloc[:, q])
					cache[(p, q)] = pq_su

				if pq_su >= q_su:
					slist = removeElement(slist, q_idx)
				q_su, q, q_idx = getNextElement(slist, q_idx)

		p_su, p, p_idx = getNextElement(slist, p_idx)
		if not p_idx:
			break

	sbest = slist[slist[:, 2] > 0, :2]
	return sbest


# In[52]:


import pandas as pd
url = '/home/d19125691/Documents/Experiments/ontologyDCQ/onto-DCQ-FS/datasets/numeric datasets/wine/wine.data'
df = pd.read_csv(url, index_col=False)
X = df.iloc[: , 1:]
Y = df.iloc[:, 0]
t = 0.05
sub = fcbf(X,Y,t)
print(sub)


# In[53]:


print(sub)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





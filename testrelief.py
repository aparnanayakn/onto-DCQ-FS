import os
import sys
from functools import reduce
from collections import deque
from multiprocessing import Pool
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


_PY2 = sys.version_info.major == 2
_STRING_TYPES = (str, unicode) if _PY2 else (str,)


class Relief(BaseEstimator, TransformerMixin):
    """Relief algorithm implementation.

    INSTANCE PROPERTIES
    ----
    w_: Weight vector
    n_iterations: Number of times to iterate
    n_features: Number of features to keep
    n_jobs: Number of concurrent jobs to use
    categorical: Iiterable of categorical feature indices
    random_state: RandomState instance used for the fitting step

    Categorical Features
    -----
    Categorical features are treated discretely even
    if their value is a floating point number. The difference
    funciton for categorical features returns only 1 or 0,
    respectively in the case in which the two values are different
    or equal.

    Weight Vector
    -----
    The Relief algorithm (and other variations) compute a weight
    vector ranking the importance of each feature. The weights can
    then be used to choose the most important features and discard
    the rest, reducing the feature space.
    """
    def __init__(self, **kwargs):
        """Initialise a new instance of this class.

        KEYWORD ARGUMENTS
        ----
        n_iterations: Number of times to iterate (defaults to 100)
        n_features: Number of features to keep (defaults to 1)
        n_jobs: Number of concurrent jobs to use (defaults to the number of available CPUs)
        categorical: Iiterable of categorical feature indices
        random_state: Seed to set before computing the weight vector or RandomState
            instance. If none is provided, a new RandomState instance is
            initialised.
        """
        kwargs = dict(kwargs)
        self.w_ = None

        def gen_random_state(rnd_state):
            """Generate random state instance"""
            if isinstance(rnd_state, np.random.RandomState):
                return rnd_state

            return np.random.RandomState(seed=rnd_state)

        for name, default_value, convf in (
                # Param name, default param value, param conversion function
                ('categorical', (), tuple),
                ('n_jobs', 1, int),
                ('n_iterations', 100, int),
                ('n_features', 1, int),
                ('random_state', None, gen_random_state)
        ):
            setattr(self, name, convf(kwargs.setdefault(name, default_value)))
            del kwargs[name]

        if self.n_jobs < 1:
            raise ValueError('n_jobs must be greater than 0')

        if kwargs:
            raise ValueError('Invalid arguments: %s' % ', '.join(kwargs))

    def fit(self, data, y):
        """Compute feature weights.

        ARGUMENTS
        ----
        data: The input data matrix
        y: The label vector
        """
        n, m = data.shape # Number of instances & features

        # Initialise state
        js = self.random_state.randint(n, size=self.n_iterations)

        # Compute weights
        self.w_ = self._fit_iteration(data, y, 0, self.n_iterations, js)

        self.w_ /= self.n_iterations

        return self

    def _fit_iteration(self, data, y, iter_offset, n_iters, js):
        w = np.array([0.] * data.shape[1])

        for i in range(iter_offset, n_iters + iter_offset):
            j = js[i]
            ri = data[j] # Random sample instance
            hit, miss = self._nn(data, y, j)

            w += np.array([
                self._diff(k, ri[k], miss[k])
                - self._diff(k, ri[k], hit[k])
                for k in range(data.shape[1])
            ])

        return w

    def _nn(self, data, y, j):
        """Return nearest instances from `data` (not) belonging to the `j`-th class.

        ARGUMENTS
        -----
        data: A numpy array
        y: The numpy array of boolean labels
        j: The index of the element to find the nearest neighbors of

        RETURN VALUE
        -----
        A tuple (h, m) of the nearest hits and misses from `data`.
        """
        ri = data[j]
        d = np.sum(
            np.array([
                self._diff(c, ri[c], data[:, c]) for c in range(len(ri))
            ]).T,
            axis=1
        )

        odata = data[d.argsort()]
        oy = y[d.argsort()]

        h = odata[oy == y[j]][0:1]
        m = odata[oy != y[j]][0]

        h = h[1] if h.shape[0] > 1 else h[0]

        return h, m

    def _diff(self, c, a1, a2):
        """Return the difference between the same attribute of two instances.

        ATTRIBUTES
        -----
        c: Feature index of the compared values
        a1: Element of the first instance to compare
        a2: Element of the second instance to compare

        RETURN VALUE
        -----
        A value in the range 0..1, where 0 means the values are the same and
        1 means they are maximally distant.
        """
        return (

            np.abs(a1 - a2) if c not in self.categorical
            else 1 - (a1 == a2)
        )

    def transform(self, data):
        """Transform the input data.

        This method uses the computed weight vector to produce a new
        dataset exhibiting only the `self.n_features` best-ranked features.

        ARGUMENTS
        -----
        data: The input data to transform

        RETURN VALUE
        -----
        A matrix with the same number of rows as `data` and at most
        `self.n_features` columns.
        """
        n_features = np.round(
            data.shape[1] * self.n_features
        ).astype(np.int16) if self.n_features < 1 else self.n_features
        feat_indices = np.flip(np.argsort(self.w_), 0)[0:n_features]

        return data[:, feat_indices]


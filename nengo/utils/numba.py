"""
Extra functions to extend the capabilities of Numba.
"""

from __future__ import absolute_import

import numpy as np

from numba import njit


@njit
def clip(x, a, b):
    """Numba-compiled version of np.clip."""
    # np.clip is not supported by numba
    # https://github.com/lmcinnes/umap/issues/84
    # note: the out option is not supported here
    y = np.empty_like(x)
    for i in range(len(x)):
        if x[i] < a:
            y[i] = a
        elif x[i] > b:
            y[i] = b
        else:
            y[i] = x[i]
    return y

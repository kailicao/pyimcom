"""This is a helper script to run Cholesky decomposition as a subprocess.
It is useful on some platforms to optimize memory usage.
"""

import sys
import numpy as np
import scipy

A = np.load(sys.argv[1])
try:
    L = scipy.linalg.cholesky(A, lower=True, check_finite=False)
except:
    L = np.zeros_like(A)
with open(sys.argv[1], 'wb') as f: np.save(f, L)


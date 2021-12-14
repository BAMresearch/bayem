import numpy as np
import tripy.utils
import pytest

N = 5
x = np.linspace(0, 1, N)
l_corr = 0.2


def correlation_func(d):
    return tripy.utils.correlation_function(
        d=d, correlation_length=l_corr, function_type="exponential"
    )


def inv_corr(x, l, N=1):
    from scipy.sparse import diags

    C0, C1 = tripy.utils.inv_cov_vec_1D(x, l, np.ones_like(x))

    CN0 = np.tile(C0, N)
    CN1 = np.zeros(len(CN0) - 1)

    for i in range(N):
        CN1[i * len(x) : (i + 1) * len(x) - 1] = C1

    return diags([CN1, CN0, CN1], [-1, 0, 1])


# Case 1: one sensor with correlated signals in time
C = tripy.utils.correlation_matrix(x.reshape(-1, 1), correlation_func)
Cinv = inv_corr(x, l_corr)

np.testing.assert_array_almost_equal(np.linalg.inv(C), Cinv.todense())


# Case 2: two sensor, each correlated in time but not beween each other
Z = np.zeros((N, N))
CC = np.block([[C, Z], [Z, C]])

CCinv = inv_corr(x, l_corr, N=2)
np.testing.assert_array_almost_equal(np.linalg.inv(CC), CCinv.todense())

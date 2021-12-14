import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import diags
from scipy.sparse.linalg import splu


def correlation_matrix(x, l, N_blocks=1):
    """
    Builds a dense correlation matrix C with exponential kernel. For `N_blocks > 1`,
    a block matrix is returned with the only C as diagonal blocks.

    x:
        1D array of locations
    l:
        correlation length
    N_blocks:
        number of uncorrelated blocks
    """
    loc = np.atleast_1d(x)
    assert len(loc.shape) == 1
    c0 = np.repeat([x], len(x), axis=0)
    r = c0 - c0.T
    C = np.exp(-abs(r) / l)

    diagonal_blocks = [C] * N_blocks
    return block_diag(*diagonal_blocks)


def inv_correlation_matrix(x, l, N_blocks=1, return_sparse=True):
    """
    Calculates the inverse of an exponential correlation matrix analytically.
    This inverse has a tridiagonal structure and is taken from [1].

    x:
        1D array of locations
    l:
        correlation length
    N_blocks:
        number of uncorrelated blocks
    return_sparse:
        If true, returns a sparse matrix with the results. If false, returns a
        tuple (C_0, C_1) with
            C_0: Main diagonal vector of inverse covariance matrix
            C_1: Off diagonal vector of inverse covariance matrix

    References:
        [1] Parameter Estimation for the Spatial Ornstein-Uhlenbeck
        Process with Missing Observations
        https://dc.uwm.edu/cgi/viewcontent.cgi?article=2131&context=etd
    """

    Nx = len(x)
    a = np.exp(-np.diff(x) / l)

    # Initialize arrays
    C_0 = np.zeros(Nx)

    # Diagonal and off diagonal elements
    a11 = 1 / (1 - a[0] ** 2)
    ann = 1 / (1 - a[-1] ** 2)
    aii = 1 / (1 - a[:-1] ** 2) + 1 / (1 - a[1:] ** 2) - 1

    # Assemble the diagonal vectors
    C_0[0] = a11
    C_0[1:-1] = aii
    C_0[-1] = ann
    C_1 = -a / (1 - a ** 2)


    CN0 = np.tile(C_0, N_blocks)
    CN1 = np.zeros(len(CN0) - 1)
    for i in range(N_blocks):
        CN1[i * Nx : (i + 1) * Nx - 1] = C_1

    if return_sparse:
        return diags([CN1, CN0, CN1], [-1, 0, 1]).tocsc()
    else:
        return C_0, C_1

def sp_logdet(M):
    """
    Computes the logdet of sparse matrix M using a LU decomposition, see
    https://stackoverflow.com/a/60982033
    """
    lu = splu(M)
    diagL = lu.L.diagonal()
    diagU = lu.U.diagonal()
    return np.log(diagL).sum() + np.log(diagU).sum()

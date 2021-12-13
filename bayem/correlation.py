"""
Created on Mon Dec 13 2021

@author: ajafari
"""

import numpy as np
# from numba import jit
# from scipy.linalg.blas import dgemm, dsyrk, dtrsm
# from scipy.linalg.lapack import dpotrf, dpttrf
from scipy.spatial.distance import pdist, squareform


def correlation_matrix(x_mx, correlation_func, distance_metric="euclidean"):
    """
    Statistical correlation matrix between measurement points based on their distance.

    Args:
        x_mx: the coordinates of the measurement points between which the correlation
            matrix is calculated, (N,K).
        correlation_func: a callable for the kernel function that calculates the statistical correlation
            between measurements made at two points that are `d` units distant from
            each other, r_major^m -> r_major^m.
        distance_metric: The distance metric to use. The distance function is the
            same as the `metric` input argument of `scipy.spatial.distance.pdist`.

    Returns:
        rho_mx: correlation matrix.
    """
    if len(x_mx.shape)!=2:
        raise ValueError('The input coordinates x_mx must be a 2-D array.')
    d = squareform(pdist(x_mx, metric=distance_metric))
    rho_mx = correlation_func(d)

    return rho_mx

class CorrelationFunction:
    def __init__(self, correlation_length, function_type="exponential", exponent=2):
        """
        Statistical correlation between measurements made at two points that are at `d`
        units distance from each other.
        Args:
            correlation_length: `1/correlation_length` controls the strength of
                correlation between two points. `correlation_length = 0` => complete
                independence for all point pairs (`rho=0`). `correlation_length = Inf` =>
                full dependence for all point pairs (`rho=1`).
            function_type: name of the correlation function.
            exponent: exponent in the type="cauchy" function. The larger the exponent the
                less correlated two points are.
        """
        if correlation_length < 0:
            raise ValueError("correlation_length must be a non-negative number.")
        self.correlation_length = correlation_length
        self.function_type = function_type.lower()
        self.exponent = exponent
    
    def __call__(self, d):
        """
        d: distance(s) between points.
        Returns:
            rho: correlation coefficient for each element of `d`.
        """
        if np.any(d < 0):
            raise ValueError("All elements of d must be non-negative numbers.")
    
        if self.correlation_length == 0:
            idx = d == 0
            rho = np.zeros(d.shape)
            rho[idx] = 1
        elif self.correlation_length == np.inf:
            rho = np.ones(d.shape)
        else:
            if self.function_type == "exponential":
                rho = np.exp(-d / self.correlation_length)
            elif self.function_type == "cauchy":
                rho = (1 + (d / self.correlation_length) ** 2) ** -self.exponent
            elif self.function_type == "gaussian":
                rho = np.exp(-((d / self.correlation_length) ** 2))
            else:
                raise ValueError(
                    "Unknown function_type. It must be one of these: 'exponential', "
                    "'cauchy', 'gaussian'."
                )
        return rho

def inv_cov_vec_1D(coord_x, l_corr, _return_sparse=True, N_blocks=1, std=None):
    """
    Calculates diagonal and off diagonal vectors of the tridiagonal inverse exponential
    covariance matrix

    Utility function for 2D loglikelihood evaluation. Given vectors of space
    and time points and the exponential covariance parameters returns the
    diagonal and off diagonal vectors of the tridiagonal inverse space and time
    covariance matrices. The expressions in [1] are modified to account for
    vector standard deviations.

    Args:
        coord_x: Vector of x
        l_corr: Correlation length in x
        std: Modeling uncertainty std dev or c.o.v. in x

    Returns:
        C_0: Main diagonal vector of inverse covariance matrix
        C_1: Off diagonal vector of inverse covariance matrix

    References:
        [1] Parameter Estimation for the Spatial Ornstein-Uhlenbeck
        Process with Missing Observations
        https://dc.uwm.edu/cgi/viewcontent.cgi?article=2131&context=etd
    """

    Nx = len(coord_x)
    a = np.exp(-np.diff(coord_x) / l_corr)
    if std is None:
        std = np.ones(Nx)

    # Initialize arrays
    C_0 = np.zeros(Nx)

    # Diagonal and off diagonal elements
    a11 = (1 / (1 - a[0] ** 2)) / std[0] ** 2
    ann = (1 / (1 - a[-1] ** 2)) / std[-1] ** 2
    aii = (1 / (1 - a[:-1] ** 2) + 1 / (1 - a[1:] ** 2) - 1) / std[1:-1] ** 2

    # Assemble the diagonal vectors
    C_0[0] = a11
    C_0[1:-1] = aii
    C_0[-1] = ann
    C_1 = (-a / (1 - a ** 2)) / (std[:-1] * std[1:])
    
    if _return_sparse:
        CN0 = np.tile(C_0, N_blocks)
        CN1 = np.zeros(len(CN0) - 1)
        for i in range(N_blocks):
            CN1[i * Nx : (i + 1) * Nx - 1] = C_1
        from scipy.sparse import diags
        return diags([CN1, CN0, CN1], [-1, 0, 1])
    else:
        return C_0, C_1

if __name__ == "__main__":
    N = 100
    L = 2
    xs = np.linspace(1, L, N).reshape((N,1))
    correlation_level = 3
    correlation_length = correlation_level*L/N
    kernel = CorrelationFunction(correlation_length=correlation_length)
    M = correlation_matrix(xs, correlation_func=kernel)
    M_inv = inv_cov_vec_1D(xs.flatten(), correlation_length)
    
    assert np.linalg.norm(M_inv@M - np.eye(N)) < 1e-10
    
    

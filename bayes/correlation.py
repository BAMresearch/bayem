import numpy as np
import logging

logger = logging.getLogger(__name__)


def transformation_SQ(locations, correlation_length, tol=1e-8):
    """
    Transformation matrix based on a squared exponential correlation matrix.
    """
    return transformation_from_correlation(
        squared_exponential(locations, correlation_length), tol
    )


def squared_exponential(locations, correlation_length):
    """
    Builds a dense correlation matrix assuming the `locations` are 
    correlated by `correlation_length`. Note that the reference to 
    distance-like names should not prevent you from putting in 
    `times` and something like a `correlation_duration` 
    """
    loc = np.atleast_1d(locations)
    assert len(loc.shape) == 1
    c0 = np.repeat([loc], len(loc), axis=0)
    r = c0 - c0.T
    return np.exp(-r * r / (2.0 * correlation_length * correlation_length))


def transformation_from_correlation(correlation, tol=1e-8):
    """
    Decompose the covariance matrix into its principal components
    Only keep the eigenvalues e with e > tol * largest eigenvalue

    Return the diagonal entries (representing the squares of the std_dev
    of independent random variables) and the corresponding eigenvectors  

    The full (correlated) sample vector X is then given by
    X = sum_i Phi_i * X_red,i with X_red,i being normal distributed with 
    zero mean and sigma2 given by the eigenvalues of the covariance matrix and Phi_i
    the corresponding eigenvalues
    """
    eigenvalues, eigen_vectors = np.linalg.eigh(correlation)
    threshold = tol * eigenvalues[-1]
    reduced_eigenvalues = eigenvalues[eigenvalues > threshold]
    reduced_eigenvectors = eigen_vectors[:, eigenvalues > threshold]
    transformation = np.divide(reduced_eigenvectors, np.sqrt(reduced_eigenvalues))
    logger.info(f"Transformation shape: {transformation.shape}")

    logger.debug(f"full eigenvalues:\n{eigenvalues}")
    logger.debug(f"threshold, {threshold}")
    logger.debug(f"reduced eigenvalues:\n{reduced_eigenvalues}")
    return transformation

"""
Created on Mon Dec 13 2021

@author: ajafari
"""
import unittest
import numpy as np
import bayem

class TestCorrelation(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_inverting_exponential_correlation_1D(self):
        N = 100
        L = 2
        xs = np.linspace(1, L, N).reshape((N,1))
        correlation_level = 3
        correlation_length = correlation_level*L/N
        kernel = bayem.CorrelationFunction(correlation_length=correlation_length)
        # Case 1: one sensor with correlated signals in time
        C = bayem.correlation_matrix(xs, correlation_func=kernel)
        C_inv = bayem.inv_cov_vec_1D(xs.flatten(), correlation_length)
        np.testing.assert_array_almost_equal(C_inv@C, np.eye(N))
        # Case 2: two sensor, each correlated in time but not beween each other
        Z = np.zeros((N, N))
        CC = np.block([[C, Z], [Z, C]])
        CCinv = bayem.inv_cov_vec_1D(xs.flatten(), correlation_length, N_blocks=2)
        np.testing.assert_array_almost_equal(CC@CCinv, np.eye(2*N))
        
if __name__ == "__main__":
    unittest.main()
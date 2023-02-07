import numpy as np
import unittest
import bayem.correlation as bc

N = 100
l = 2
xs = np.linspace(1, l, N)
correlation_length = 0.5
C = bc.cor_exp_1d(xs, correlation_length)


class TestCorrelation(unittest.TestCase):
    def test_single_sensor(self):
        C_inv = bc.inv_cor_exp_1d(xs, correlation_length)
        np.testing.assert_array_almost_equal(C_inv @ C, np.eye(N))
    
    def test_two_sensors(self):
    
        Z = np.zeros((N, N))
        CC = np.block([[C, Z], [Z, C]])
    
        np.testing.assert_array_almost_equal(
            CC, bc.cor_exp_1d(xs, correlation_length, N_blocks=2)
        )
    
        CCinv = bc.inv_cor_exp_1d(xs, correlation_length, N_blocks=2)
        np.testing.assert_array_almost_equal(CC @ CCinv, np.eye(2 * N))
    
    def test_logdet(self):
        from scipy.sparse.linalg import splu
    
        C_inv = bc.inv_cor_exp_1d(xs, l)
        logdet_numpy = np.linalg.slogdet(C_inv.todense())[1]
        logdet_lu = bc.sp_logdet(C_inv)
    
        self.assertAlmostEqual(logdet_numpy, logdet_lu, delta = 1e-6 * logdet_numpy)

if __name__ == "__main__":
    unittest.main()

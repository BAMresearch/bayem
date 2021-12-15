import numpy as np
import bayem
import pytest

N = 100
l = 2
xs = np.linspace(1, l, N)
correlation_length = 0.5
C = bayem.cor_exp_1d(xs, correlation_length)


def test_single_sensor():
    C_inv = bayem.inv_cor_exp_1d(xs, correlation_length)
    np.testing.assert_array_almost_equal(C_inv @ C, np.eye(N))


def test_two_sensors():

    Z = np.zeros((N, N))
    CC = np.block([[C, Z], [Z, C]])

    np.testing.assert_array_almost_equal(
        CC, bayem.cor_exp_1d(xs, correlation_length, N_blocks=2)
    )

    CCinv = bayem.inv_cor_exp_1d(xs, correlation_length, N_blocks=2)
    np.testing.assert_array_almost_equal(CC @ CCinv, np.eye(2 * N))


def test_logdet():
    from scipy.sparse.linalg import splu

    C_inv = bayem.inv_cor_exp_1d(xs, l)
    logdet_numpy = np.linalg.slogdet(C_inv.todense())[1]
    logdet_lu = bayem.sp_logdet(C_inv)

    assert logdet_numpy == pytest.approx(logdet_lu)


if __name__ == "__main__":
    test_single_sensor()
    test_two_sensors()
    test_logdet()

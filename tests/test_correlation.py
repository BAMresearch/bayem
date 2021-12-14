import numpy as np
import bayem


N = 100
L = 2
xs = np.linspace(1, L, N)
correlation_length = 0.5
C = bayem.correlation_matrix(xs, correlation_length)


def test_single_sensor():
    C_inv = bayem.inv_correlation_matrix(xs, correlation_length)
    np.testing.assert_array_almost_equal(C_inv @ C, np.eye(N))


def test_two_sensors():

    Z = np.zeros((N, N))
    CC = np.block([[C, Z], [Z, C]])

    np.testing.assert_array_almost_equal(
        CC, bayem.correlation_matrix(xs, correlation_length, N_blocks=2)
    )

    CCinv = bayem.inv_correlation_matrix(xs, correlation_length, N_blocks=2)
    np.testing.assert_array_almost_equal(CC @ CCinv, np.eye(2 * N))


if __name__ == "__main__":
    test_single_sensor()
    test_two_sensors()

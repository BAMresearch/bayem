import bayem
import numpy as np
import pytest

np.random.seed(6174)
x = np.linspace(0, 1, 1000)
A, B, sd = 7.0, 42.0, 0.1
data = A * x + B + np.random.normal(0, sd, len(x))


def model_error(parameters):
    return parameters[0] * x + parameters[1] - data


x0 = bayem.MVN([6, 11], [[1 / 3 ** 2, 0], [0, 1 / 3 ** 2]])
info = bayem.vba(model_error, x0, noise0=None)
print(info)


def test_results():
    for i, correct_value in enumerate([A, B]):
        posterior_mean = info.param.mean[i]
        posterior_std = info.param.std_diag[i]

        assert posterior_std < 0.3
        assert posterior_mean == pytest.approx(correct_value, abs=2 * posterior_std)

    post_noise_precision = info.noise.mean
    post_noise_std = 1.0 / post_noise_precision ** 0.5
    assert post_noise_std == pytest.approx(sd, rel=0.01)

    assert (
        info.nit < 4
    )  # For such linear regression we do expect to have very few iterations.
    assert info.t < 0.1

    info.summary(True)
    info.summary(tablefmt="fancy_grid")

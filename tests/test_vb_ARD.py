import numpy as np
import pytest
import bayem
from test_vb import ModelError

import logging

# logging.basicConfig(level=logging.INFO)

"""
Test having an ARD parameter (see Chappell paper for more info)
True model (used for data generation) is defined as a linear model + bias term at each "sensor" location (xs)
Bias term represents how the fw model deviates from the true model and should be inferred during VB. Since the bias parameters are sparse, an ARD prior is set for them.
"""


class ForwardModel:
    def __init__(self):
        self.xs = np.linspace(1.0, 10, 20)

    def __call__(self, parameters):
        m = parameters[0]
        b = parameters[1:]
        return self.xs * m + b

    def jacobian(self, parameters):
        b = parameters[1:]

        d_dm = self.xs
        all_d = [d_dm]
        for i in range(len(b)):
            v = np.zeros_like(self.xs)
            v[i] = 1
            all_d.append(v)

        return np.array(all_d).T


np.random.seed(6174)

fw = ForwardModel()
n_sensors = len(fw.xs)

param_true = np.zeros(1 + n_sensors)
param_true[0] = 7.0
param_true[6] = 1  # ARD!

noise_std = 0.1

data = []
perfect_data = fw(param_true)
for _ in range(10):
    data.append(perfect_data + np.random.normal(0, noise_std, len(perfect_data)))

me = ModelError(fw, data, with_jacobian=False)

# setting mean and precision
prec = np.identity(len(param_true))
prec[0, 0] = 1 / 3 ** 2
prec[1, 1] = 1 / 3 ** 2
for i in range(n_sensors):
    prec[1 + i, 1 + i] = 2
param_prior = bayem.MVN(mean=np.array([6] + [0] * n_sensors), precision=prec)
noise_prior = bayem.Gamma.FromSDQuantiles(0.1 * noise_std, 10 * noise_std)

info = bayem.vba(
    me,
    param_prior,
    noise_prior,
    jac=False,
    index_ARD=np.arange(1, n_sensors + 1),
    maxiter=500,
    maxtrials=50,
    update_noise=True,
)


def test_checks():
    means, sds = info.param.mean, info.param.std_diag
    for i, p in enumerate(param_true):
        mean, sd = means[i], sds[i]
        assert sd < 0.05
        assert mean == pytest.approx(p, abs=2 * sd)

    post_noise_precision = info.noise.mean
    post_noise_std = 1.0 / post_noise_precision ** 0.5
    assert post_noise_std == pytest.approx(noise_std, rel=0.01)

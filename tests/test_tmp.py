import numpy as np
import bayem
import pytest

np.random.seed(6174)
xs = np.linspace(0.01, 0.1, 100)

param_true = [7.0, 10.0]
noise_sd = 0.1
data = param_true[1] + xs * param_true[0] + np.random.normal(0, noise_sd, size=xs.shape)


def f(x):
    m, c = x
    return {"noise": c + xs * m - data}

def jac(x):
    d_dm = xs
    d_dc = np.ones_like(xs)
    return {"noise": np.vstack([d_dm, d_dc]).T}


x0 = bayem.MVN([0, 11], [[1 / 7 ** 2, 0], [0, 1 / 3 ** 2]])
noise0 = { "noise": bayem.Gamma.FromSDQuantiles(0.5 * noise_sd, 1.5 * noise_sd) }
info = bayem.vba(f, x0, noise0, jac)

print(info)



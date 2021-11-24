import numpy as np
import bayem

np.random.seed(6174)
xs = np.linspace(0.01, 0.1, 100)

param_true = [7.0, 10.0]
noise_sd = 0.1
data = param_true[1] + xs * param_true[0] + np.random.normal(0, noise_sd, size=xs.shape)


# [1] Define the problem in terms of numpy arrays
def f(x):
    m, c = x
    return c + xs * m - data


def jac(x):
    d_dm = xs
    d_dc = np.ones_like(xs)
    return np.vstack([d_dm, d_dc]).T


n0 = bayem.Gamma.FromSDQuantiles(0.5 * noise_sd, 1.5 * noise_sd)

# [2] Define the problem as list of numpy arrays


def f_list(x):
    return [f(x)]


def jac_list(x):
    return [jac(x)]


n0_list = [n0]

# [3] Define the problem as dict of numpy arrays


def f_dict(x):
    return {"noise": f(x)}


def jac_dict(x):
    return {"noise": jac(x)}


n0_dict = {"noise": n0}


# [4] Define f to return both k and jac
def f_jac(x):
    return f(x), jac(x)


# [5] Define f to return both k and jac as dict
def f_jac_dict(x):
    return f_dict(x), jac_dict(x)


x0 = bayem.MVN([0, 11], [[1 / 7 ** 2, 0], [0, 1 / 3 ** 2]])


def test_provide_everything():
    assert bayem.vba(f, x0, n0, jac).success
    assert bayem.vba(f_list, x0, n0_list, jac_list).success
    assert bayem.vba(f_dict, x0, n0_dict, jac_dict).success
    assert bayem.vba(f_jac, x0, n0, jac=True).success
    assert bayem.vba(f_jac_dict, x0, n0_dict, jac=True).success


def test_no_noise():
    assert bayem.vba(f, x0, jac=jac).success
    assert bayem.vba(f_list, x0, jac=jac_list).success
    assert bayem.vba(f_dict, x0, jac=jac_dict).success
    assert bayem.vba(f_jac, x0, jac=True).success


def test_no_jacobian():
    assert bayem.vba(f, x0, n0).success
    assert bayem.vba(f_list, x0, n0_list).success
    assert bayem.vba(f_dict, x0, n0_dict).success


def test_minimal():
    assert bayem.vba(f, x0).success
    assert bayem.vba(f_list, x0).success
    assert bayem.vba(f_dict, x0).success


def test_returned_noise_type():
    assert isinstance(bayem.vba(f, x0).noise, bayem.Gamma)
    assert isinstance(bayem.vba(f_list, x0).noise, list)
    assert isinstance(bayem.vba(f_dict, x0).noise, dict)

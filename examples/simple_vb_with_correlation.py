import numpy as np
import matplotlib.pyplot as plt

import bayem.vba as vba
import bayem.distributions as bd
import bayem.correlation as bc

def g(theta):
    return theta[1] ** 2 + xs * theta[0]

class F:
    def __init__(self, data):
        self.data = data

    def __call__(self, theta):
        k = g(theta) - self.data
        d_dm = xs
        d_dc = 2 * theta[1] * np.ones_like(xs)
        return k, np.vstack([d_dm, d_dc]).T

param = [5, 7]
param_prec = 0.001
N = 100
L = 2
xs = np.linspace(0, L, N)

noise_std = 0.2
correlation_level = 5
cor_length = correlation_level * L / N
noise_cov = bc.cor_exp_1d(xs, cor_length) * noise_std ** 2
cov_inv = bc.inv_cor_exp_1d(xs, cor_length)

perfect_data = g(param)
np.random.seed(6174)
correlated_noise = np.random.multivariate_normal(
    np.zeros(len(xs)),
    noise_cov ,
)
correlated_data = perfect_data + correlated_noise
plt.figure()
plt.plot(perfect_data, label="perfect", marker="*", linestyle="")
plt.plot(correlated_data, label="noisy", marker="+", linestyle="")
plt.title("Data")
plt.legend()
plt.show()

f = F(correlated_data)

def do_vb(cov_inv=None):
    m0 = np.array([2, 19])
    L0 = np.array([[param_prec, 0], [0, param_prec]])
    prior_mvn = bd.MVN(m0, L0)
    
    cov_log_det = None if cov_inv is None else (-bc.sp_logdet(cov_inv))
    noise0 = bd.Gamma(shape=1e-6, scale=1e6)
    vb_results = vba(f=f, x0=prior_mvn, noise0=noise0, cov_inv=cov_inv, cov_log_det=cov_log_det \
                     , jac=True, update_noise=True, maxiter=100, tolerance=1e-8, store_full_precision=True)
    
    noise_prec_mean = vb_results.noise.shape * vb_results.noise.scale
    aa = 'without' if cov_inv is None else 'with'
    _msg = f"\n----- VB {aa} covariance of noise:"
    _msg += f"\n\t- Max. free energy = {vb_results.f_max:.2f} ,"
    _msg += f"\n\t- Target pars = {param} ,"
    _msg += f"\n\t- Inferred pars (mean) = {vb_results.param.mean} ,"
    _msg += f"\n\t- Inferred pars (precision) = {vb_results.param.precision} ,"
    _msg += f"\n\t- Target noise precision = {noise_std ** (-2):.2f} ,"
    _msg += f"\n\t- Inferred noise precision (mean): {noise_prec_mean:.2f} ."
    print(_msg)
    
    return vb_results

def plot_posteriors(vb_resultss, labels, colors):
    from scipy.stats import norm
    
    # same prior for all inferences
    m0 = vb_resultss[0].param0.mean
    stds0 = vb_resultss[0].param0.std_diag
    
    stds1 = []
    stds2 = []
    means1 = []
    means2 = []
    for vb_results in vb_resultss:
        stds1.append(vb_results.param.precision[0, 0] ** (-0.5))
        stds2.append(vb_results.param.precision[1, 1] ** (-0.5))
        means1.append(vb_results.param.mean[0])
        means2.append(vb_results.param.mean[1])
    stds1_max = max(stds1)
    stds2_max = max(stds2)
    _min1 = min(param[0], min(means1))
    _max1 = max(param[0], max(means1))
    _min2 = min(param[1], min(means2))
    _max2 = max(param[1], max(means2))
    xs1 = np.linspace(_min1 - 3 * stds1_max, _max1 + 3 * stds1_max, 1000)
    xs2 = np.linspace(_min2 - 3 * stds2_max, _max2 + 3 * stds2_max, 1000)
    plt.figure()
    dis0 = norm(m0[0], stds0[0])
    pdfs0 = dis0.pdf(xs1)
    plt.plot(xs1, pdfs0, label='prior', color='gray')
    for i, vb_results in enumerate(vb_resultss):
        dis1 = norm(means1[i], stds1[i])
        pdfs1 = dis1.pdf(xs1)
        plt.plot(xs1, pdfs1, label=labels[i], color=colors[i])
    plt.axvline(x=param[0], ls="-", label='target', color='black')
    plt.legend()
    plt.title("Parameter 1")

    plt.figure()
    dis0 = norm(m0[1], stds0[1])
    pdfs0 = dis0.pdf(xs2)
    plt.plot(xs2, pdfs0, label='prior', color='gray')
    for i, vb_results in enumerate(vb_resultss):
        dis2 = norm(means2[i], stds2[i])
        pdfs2 = dis2.pdf(xs2)
        plt.plot(xs2, pdfs2, label=labels[i], color=colors[i])
    plt.axvline(x=param[1], ls="-", label='target', color='black')
    plt.legend()
    plt.title("Parameter 2")

    plt.show()


if __name__ == "__main__":
    vb_resultss = []
    labels = []
    colors = []

    vb_results = do_vb()
    vb_resultss.append(vb_results)
    labels.append("post. without Cor. matrix")
    colors.append('orange')

    vb_results2 = do_vb(cov_inv=cov_inv)
    vb_resultss.append(vb_results2)
    labels.append("post. with target Cor. matrix")
    colors.append('green')

    _factor = 1 / 2
    vb_results3 = do_vb(cov_inv=cov_inv / _factor)
    vb_resultss.append(vb_results3)
    labels.append(f"post. with {_factor} * target Cor. matrix")
    colors.append('blue')

    # In the first scenario we obtain more certain (higher precision) inference of parameters
    # , since we did not account for correlation of data.
    assert vb_results.param.precision[0, 0] > vb_results2.param.precision[0, 0]
    assert vb_results.param.precision[1, 1] > vb_results2.param.precision[1, 1]

    ## Since correlation matrix in info3 is half of info2 (only a scaling):
    # 1) The inferred parameters in info2 and info3 must be the same.
    # 2) The inferred noise precision in info3 must be half of info2.
    # This ratio, however, only goes to the mean of the identified noise, meaning that:
    # The scales are different by the factor 2.
    # But the shapes are the same (we only have shift of one distribution from the other).
    # 3) The converged free energy in info2 and info3 is the same.
    assert (
        np.linalg.norm(vb_results3.param.mean - vb_results2.param.mean) / np.linalg.norm(vb_results2.param.mean)
        < 1e-6
    )
    assert (
        np.linalg.norm(vb_results3.param.precision - vb_results2.param.precision)
        / np.linalg.norm(vb_results2.param.precision)
        < 1e-6
    )
    assert abs((vb_results3.noise.shape - vb_results2.noise.shape) / vb_results2.noise.shape) < 1e-6
    assert abs((vb_results3.noise.scale - _factor * vb_results2.noise.scale) / vb_results2.noise.scale) < 1e-6
    assert abs((vb_results3.f_max - vb_results2.f_max) / vb_results2.f_max) < 1e-5

    plot_posteriors(vb_resultss, labels, colors)

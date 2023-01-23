import numpy as np
import scipy.special as special
import scipy.linalg
import matplotlib.pyplot as plt

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
perfect_data = g(param)
noise_std = 0.2

correlation_level = 5
noise_cor_matrix = bc.cor_exp_1d(xs, correlation_level * L / N)

Cinv = bc.inv_cor_exp_1d(xs, correlation_level * L / N)


def free_energy(m, m0, L, L0, L_inv, s, s0, c, c0, k, J, C_inv, C_inv_logdet):
    f_new = -0.5 * ((m - m0).T @ L0 @ (m - m0) + np.trace(L_inv @ L0))
    sign, logdet = np.linalg.slogdet(L)
    f_new -= 0.5 * sign * logdet

    f_new += 0.5 * C_inv_logdet
    f_new += 0.5 * len(m)

    sign0, logdet0 = np.linalg.slogdet(L0)
    f_new += 0.5 * sign0 * logdet0

    N = len(k)

    # From the update equation
    f_new += -s * c / s0 + (N / 2 + c0 - 1) * (np.log(s) + special.digamma(c))
    f_new += -0.5 * s * c * (k.T @ C_inv @ k + np.trace(L_inv @ J.T @ C_inv @ J))
    f_new += c * np.log(s) + special.gammaln(c)
    f_new += c - (c - 1) * (np.log(s) + special.digamma(c))
    # constant terms to fix the evidence
    f_new += -N / 2 * np.log(2 * np.pi) - special.gammaln(c0) - c0 * np.log(s0)
    return f_new


def vba(f, m0, L0, s0=1e6, c0=1e-6, C_inv=None, update_noise=True):
    m = np.copy(m0)
    L = np.array(L0)

    k, J = f(m)

    if C_inv is None:
        C_inv = scipy.sparse.identity(len(k))

    C_inv_logdet = bc.sp_logdet(C_inv)

    s = np.copy(s0)
    c = np.copy(c0)

    f_old = -np.inf

    i_iter = 0
    while True:
        i_iter += 1

        # update prm
        L = s * c * J.T @ C_inv @ J + L0
        L_inv = np.linalg.inv(L)
        Lm = s * c * J.T @ C_inv @ (-k + J @ m) + L0 @ m0
        m = Lm @ L_inv

        k, J = f(m)

        # update noise
        if update_noise:
            c = len(k) / 2 + c0
            s_inv = 1 / s0 + 0.5 * k.T @ C_inv @ k + 0.5 * np.trace(L_inv @ J.T @ C_inv @ J)
            s = 1 / s_inv

        print(f"current mean: {m}")

        f_new = free_energy(
            m, m0, L, L0, L_inv, s, s0, c, c0, k, J, C_inv, C_inv_logdet
        )

        print(f"Free energy of iteration {i_iter} is {f_new}")

        if abs(f_old - f_new) < 1.0e-6:
            break  # sucess!

        if i_iter > 50:
            raise RuntimeError("No convergence after 50 iterations")

        f_old = f_new

    delta_f = f_old - f_new
    print(f"Stopping VB. Iterations:{i_iter}, free energy change {delta_f}.")

    return {"mean": m, "precision": L, "scale": s, "shape": c, "F": f_new}


def do_vb(_plot=True, C_inv=None):
    np.random.seed(6174)
    correlated_noise = np.random.multivariate_normal(
        np.zeros(len(xs)),
        noise_cor_matrix * noise_std ** 2,
    )

    correlated_data = perfect_data + correlated_noise
    f = F(correlated_data)

    if _plot:
        plt.figure()
        plt.plot(perfect_data, label="perfect", marker="*", linestyle="")
        plt.plot(correlated_data, label="noisy", marker="+", linestyle="")
        plt.title("Data")
        plt.legend()
        plt.show()

    m0 = np.array([2, 19])
    L0 = np.array([[param_prec, 0], [0, param_prec]])

    info = vba(f, m0, L0, C_inv=C_inv)
    for what, value in info.items():
        print(what, value)
    noise_prec_mean = info["shape"] * info["scale"]
    print(f"Noise-Std from mean of identified precision: {noise_prec_mean ** (-0.5)} .")
    
    info['prior'] = {'m0': m0, 'L0': L0}

    return info


def plot_posteriors(infos, labels, colors):
    from scipy.stats import norm
    
    # same prior for all inferences
    m0 = infos[0]['prior']['m0']
    stds0 = np.diag(infos[0]['prior']['L0'] ** (-0.5))
    
    stds1 = []
    stds2 = []
    means1 = []
    means2 = []
    for info in infos:
        stds1.append(info["precision"][0, 0] ** (-0.5))
        stds2.append(info["precision"][1, 1] ** (-0.5))
        means1.append(info["mean"][0])
        means2.append(info["mean"][1])
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
    for i, info in enumerate(infos):
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
    for i, info in enumerate(infos):
        dis2 = norm(means2[i], stds2[i])
        pdfs2 = dis2.pdf(xs2)
        plt.plot(xs2, pdfs2, label=labels[i], color=colors[i])
    plt.axvline(x=param[1], ls="-", label='target', color='black')
    plt.legend()
    plt.title("Parameter 2")

    plt.show()


if __name__ == "__main__":
    infos = []
    labels = []
    colors = []

    info = do_vb()
    infos.append(info)
    labels.append("post. without Cor. matrix")
    colors.append('orange')

    info2 = do_vb(C_inv=Cinv)
    infos.append(info2)
    labels.append("post. with target Cor. matrix")
    colors.append('green')

    _factor = 1 / 2
    info3 = do_vb(C_inv=Cinv / _factor)
    infos.append(info3)
    labels.append(f"post. with {_factor} * target Cor. matrix")
    colors.append('blue')

    # In the first scenario we obtain more certain (higher precision) inference of parameters
    # , since we did not account for correlation of data.
    assert info["precision"][0, 0] > info2["precision"][0, 0]
    assert info["precision"][1, 1] > info2["precision"][1, 1]

    ## Since correlation matrix in info3 is half of info2 (only a scaling):
    # 1) The inferred parameters in info2 and info3 must be the same.
    # 2) The inferred noise precision in info3 must be half of info2.
    # This ratio, however, only goes to the mean of the identified noise, meaning that:
    # The scales are different by the factor 2.
    # But the shapes are the same (we only have shift of one distribution from the other).
    # 3) The converged free energy in info2 and info3 is the same.
    assert (
        np.linalg.norm(info3["mean"] - info2["mean"]) / np.linalg.norm(info2["mean"])
        < 1e-6
    )
    assert (
        np.linalg.norm(info3["precision"] - info2["precision"])
        / np.linalg.norm(info2["precision"])
        < 1e-6
    )
    assert abs((info3["shape"] - info2["shape"]) / info2["shape"]) < 1e-6
    assert abs((info3["scale"] - _factor * info2["scale"]) / info2["scale"]) < 1e-6
    assert abs((info3["F"] - info2["F"]) / info2["F"]) < 1e-5

    plot_posteriors(infos, labels, colors)

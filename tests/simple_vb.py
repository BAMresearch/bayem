import numpy as np
import scipy.special as special


def free_energy(m, m0, L, L0, L_inv, s, s0, c, c0, k, J):
    f_new = -0.5 * ((m - m0).T @ L0 @ (m - m0) + np.trace(L_inv @ L0))
    (sign, logdet) = np.linalg.slogdet(L)
    f_new -= 0.5 * sign * logdet
    f_new += 0.5 * len(m)

    (sign0, logdet0) = np.linalg.slogdet(L0)
    f_new += 0.5 * sign0 * logdet0

    N = len(k)

    # From the update equation
    f_new += -s * c / s0 + (N / 2 + c0 - 1) * (np.log(s) + special.digamma(c))
    f_new += -0.5 * s * c * (k.T @ k + np.trace(L_inv @ J.T @ J))
    f_new += c * np.log(s) + special.gammaln(c)
    f_new += c - (c - 1) * (np.log(s) + special.digamma(c))
    # constant terms to fix the evidence
    f_new += -N / 2 * np.log(2 * np.pi) - special.gammaln(c0) - c0 * np.log(s0)
    return f_new


def vba(f, m0, L0, s0=1e6, c0=1e-6):
    m = np.copy(m0)
    L = np.array(L0)

    k, J = f(m)

    s = np.copy(s0)
    c = np.copy(c0)

    f_old = -np.inf

    i_iter = 0
    while True:
        i_iter += 1

        # update prm
        L = s * c * J.T @ J + L0
        L_inv = np.linalg.inv(L)
        Lm = s * c * J.T @ (-k + J @ m) + L0 @ m0
        m = Lm @ L_inv

        # update noise
        c = len(k) / 2 + c0
        s_inv = 1 / s0 + 0.5 * k.T @ k + 0.5 * np.trace(L_inv @ J.T @ J)
        s = 1 / s_inv

        k, J = f(m)

        print(f"current mean: {m}")

        f_new = free_energy(m, m0, L, L0, L_inv, s, s0, c, c0, k, J)

        print(f"Free energy of iteration {i_iter} is {f_new}")

        if abs(f_old - f_new) < 1.0e-1:
            break  # sucess!

        if i_iter > 50:
            raise RuntimeError("No convergence after 50 iterations")

        f_old = f_new

    delta_f = f_old - f_new
    print(f"Stopping VB. Iterations:{i_iter}, free energy change {delta_f}.")

    return {"mean": m, "precision": L, "scale": s, "shape": c, "F": f_new}


if __name__ == "__main__":
    np.random.seed(6174)
    N = 100
    xs = np.linspace(1, 2, N)

    def g(theta):
        return theta[1] ** 2 + xs * theta[0]

    noise = 1.0
    data = g([5, 7]) + np.random.normal(0, noise, size=N)

    def f(theta):
        k = g(theta) - data

        d_dm = xs
        d_dc = 2 * theta[1] * np.ones_like(xs)
        return k, np.vstack([d_dm, d_dc]).T

    m0 = np.array([2, 19])
    L0 = np.array([[0.001, 0], [0, 0.001]])

    info = vba(f, m0, L0)
    for what, value in info.items():
        print(what, value)

    import bayem

    info = bayem.vba(f, x0=bayem.MVN(m0, L0), jac=True)
    print(1 / info.noise.mean ** 0.5)
    print(info)
    bayem.result_trace(info)

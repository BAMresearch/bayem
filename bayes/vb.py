import numpy as np
import scipy.stats
import scipy.special as special

import logging

logger = logging.getLogger(__name__)


def pretty_array(a, offset=0):
    s = np.array2string(a, max_line_width=70, formatter={"float": "{: 7.3e}".format})
    return s.replace("\n", "\n │" + " " * offset)


class MVN:
    def __init__(self, mean=[0.0], precision=[[1.0]], name="MVN"):
        self.mean = np.atleast_1d(mean).astype(float)
        self.precision = np.atleast_2d(precision).astype(float)
        self.cov = np.linalg.inv(self.precision)
        self.name = name

    @property
    def std_diag(self):
        return np.sqrt(np.diag(self.cov))

    @property
    def covariance(self):
        return self.cov

    def pdf(self, xs, i):
        """return pdf for all xs for i-th parameter"""
        return scipy.stats.norm.pdf(xs, self.mean[i], self.std_diag[i])

    def __str__(self):
        s = f"{self.name} with \n"
        s += f" ├── mean: {pretty_array(self.mean, 9)}\n"
        s += f" ├── std:  {pretty_array(self.std_diag, 9)}"
        return s


class Gamma:
    def __init__(self, s=1.0, c=1.0, name="Gamma"):
        self.s = s  # shape
        self.c = c  # scale
        self.name = name

    @property
    def mean(self):
        return self.s * self.c

    def pdf(self, xs):
        return scipy.stats.gamma.pdf(xs, a=self.s, scale=self.c)

    def __repr__(self):
        return f"{self.name} with \n └── mean: {self.mean, 9}"

    @classmethod
    def FromSD(cls, sd, shape=1.0):
        return cls(shape, 1.0 / sd ** 2 / shape)

    @classmethod
    def Noninformative(cls):
        """
        Suggested by @ilma following
        https://projecteuclid.org/euclid.ejs/1320416981
        or
        https://math.stackexchange.com/questions/449234/vague-gamma-prior
        """
        return cls(s=1.0 / 3.0, c=0.0)


def plot_pdf(
    dist,
    expected_value=None,
    name1="Posterior",
    name2="Prior",
    plot="individual",
    color_plot="r",
    **kwargs,
):
    import matplotlib.pyplot as plt

    if ("min" in kwargs) & ("max" in kwargs):
        x_min = kwargs["min"]
        x_max = kwargs["max"]
    else:
        x_min = np.min(dist.mean) * 0.02
        x_max = np.max(dist.mean) * 3
    # Create grid and multivariate normal
    x_plot = np.linspace(x_min, x_max, 5000)

    if expected_value is not None:
        expected_value = np.atleast_1d(expected_value)

    for i in range(len(dist.mean)):
        plt.plot(
            x_plot, dist.pdf(x_plot, i), label="%s of parameter nb %i" % (name1, i)
        )
        if expected_value is not None:
            plt.axvline(expected_value[i], ls=":", c="k")
        if "compare_with" in kwargs:
            plt.plot(
                x_plot,
                kwargs["compare_with"].pdf(x_plot, i),
                label="%s of parameter nb %i" % (name2, i),
            )

        plt.xlabel("Parameter")
        plt.legend()
        if plot == "individual":
            plt.show()
    if plot == "joint":
        plt.show()


class CountedEval:
    def __init__(self, f):
        self._f = f
        self.n = 0

    def __call__(self, *args, **kwargs):
        self.n += 1
        return self._f(*args, **kwargs)


class Jacobian:
    def __init__(self, f):
        self._f = f

    def __call__(self, xp):
        """
        calculate the numeric jacobian via central differences.

        We _could_ change the argument x directly. For safety, we make a copy
        at the beginning and work with that.
        """
        x = np.copy(xp)

        for iParam in range(len(x)):
            dx = x[iParam] * 1.0e-7  # approx x0 * sqrt(machine precision)
            if dx == 0:
                dx = 1.0e-10

            x[iParam] -= dx
            fs0 = self._f(x)
            x[iParam] += 2 * dx
            fs1 = self._f(x)
            x[iParam] -= dx

            if iParam == 0:
                # allocate jac
                jac = []
                for f0 in fs0:
                    jac.append(np.empty([len(f0), len(x)]))

            for i, (f0, f1) in enumerate(zip(fs0, fs1)):
                jac[i][:, iParam] = -(f1 - f0) / (2 * dx)

        return jac


class VerifiedModelError:
    """
    Brings the user-provided forward model in a algorithm-compatible form.
    * transforms a simple (numpy) vector output into a list of length 1
    * adds a CDF jacobian, if not provided
    """

    def __init__(self, model_error):
        self._f = CountedEval(model_error)
        self.n_jac = 0

        try:
            self._jac = model_error.jacobian
        except:
            self._jac = Jacobian(self.__call__)

    def __call__(self, parameters):
        f = self._f(parameters)
        if not isinstance(f, list):
            f = [f]
        return f

    def jacobian(self, parameters):
        jac = self._jac(parameters)
        if not isinstance(jac, list):
            jac = [jac]
        self.n_jac += 1
        return jac

    @property
    def n(self):
        return self._f.n


def variational_bayes(model_error, param0, noise0=None, **kwargs):
    """
    Nonlinear variational bayes algorithm according to
    @article{chappell2008variational,
          title={Variational Bayesian inference for a nonlinear forward model},
          author={Chappell, Michael A and Groves, Adrian R and Whitcher, 
                  Brandon and Woolrich, Mark W},
          journal={IEEE Transactions on Signal Processing},
          volume={57},
          number={1},
          pages={223--236},
          year={2008},
          publisher={IEEE}
        }

    This implements the formulas of section III.C (Extending the Noise Model)
    with the same notation and references to each formula, with the only 
    exception that capital lambda L in the paper is here referred to as L.

    model_error that contains
        __call__(parameter_means):
            * difference of the forward model to the data
            * list of numpy vectors where each list corresponds to one noise group
            * alternatively: just a numpy vector for the case of exactly one
                             noise group

        jacobian(parameter_means) [optional]:
            * total jacobian of the forward model w.r.t. the parameters
            * list of numpy matrices where each list corresponds to one noise group
            * alternatively: just a numpy matrix for the case of exactly one
                             noise group
    param0:
        multivariate normal distributed parameter prior

    noise0:
        list of gamma distributions for the noise prior
        If noise0 is None, a noninformative gamma prior is chosen 
        automatically according to the number of noise groups.

    tolerance:
        free energy change that causes the algorithm to stop

    iter_max:
        maximum number of iterations

    index_ARD:
        Automatic Relevance Determination option to allow for "the automated reduction of model
        complexity" (Chappell et al. 2009). Should be passed (as kwargs) when ARD is applied to
        one of the model parameters described by the MVN. It should be an array containing
        the indexes corresponding to the position of the ARD parameters in the m and dig(L)
        vectors.

    Returns:
        VBResult defined below
    """
    vb = VB()
    return vb.run(model_error, param0, noise0, **kwargs)


def variational_bayes_nonlinear(model_error, param0, noise0=None, **kwargs):
    logger.warning(
        "'variational_bayes_nonlinear' is deprecated. Use 'variational_bayes' "
        " with the same arguments that only returns a 'VBResult' instance."
    )
    vb = VB()
    result = vb.run(model_error, param0, noise0, **kwargs)
    return result.param, result.noise, result


class VBResult:
    """
    Somehow inspired by 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """

    def __init__(self):
        self.param = None
        self.noise = None
        self.success = False
        self.message = ""
        self.free_energies = []
        self.nfev = 0
        self.njev = 0
        self.nit = 0

    def __str__(self):
        s = "### VB Result ###\n"

        for member, value in vars(self).items():
            v = str(value).replace("\n", "\n" + " " * 17)
            s += f"  {member+':':14s} {v}\n"

        return s


class VB:
    def __init__(self, n_trials_max=10, iter_max=50, tolerance=0.1):
        self.f_old = -np.inf
        self.f_stored = -np.inf
        self.param_stored = None
        self.n_trials = 0
        self.n_trials_max = n_trials_max
        self.tolerance = tolerance
        self.iter_max = iter_max

        self.result = VBResult()

    def run(self, model_error, param0, noise0=None, **kwargs):

        if "tolerance" in kwargs:
            self.tolerance = kwargs["tolerance"]
        if "iter_max" in kwargs:
            self.iter_max = kwargs["iter_max"]
        if "n_trials_max" in kwargs:
            self.n_trials_max = kwargs["n_trials_max"]

        model_error = VerifiedModelError(model_error)
        k, J = model_error(param0.mean), model_error.jacobian(param0.mean)

        if noise0 is None:
            noise0 = [Gamma.Noninformative() for i in range(len(k))]

        if isinstance(noise0, Gamma):
            noise0 = [noise0]

        if len(k) != len(noise0):
            error = f"Your model error contains {len(k)} noise "
            error += f" terms, your noise prior contains {len(noise0)}."
            raise ValueError(error)

        # adapt notation
        s = [n.s for n in noise0]
        c = [n.c for n in noise0]
        m = np.copy(param0.mean)
        L = np.copy(param0.precision)

        m0 = np.copy(m)
        L0 = np.copy(L)
        s0 = np.copy(s)
        c0 = np.copy(c)

        self.param_stored = [np.copy(s), np.copy(c), np.copy(m), np.copy(L)]

        N = len(s)
        i_iter = 0
        while True:
            i_iter += 1

            # fw model parameter update
            L = sum([s[i] * c[i] * J[i].T @ J[i] for i in range(N)]) + L0
            L_inv = np.linalg.inv(L)

            Lm = sum([s[i] * c[i] * J[i].T @ (k[i] + J[i] @ m) for i in range(N)])
            Lm += L0 @ m0
            m = Lm @ L_inv

            k, J = model_error(m), model_error.jacobian(m)

            # noise parameter update
            for i in range(N):
                # formula (30)
                c[i] = len(k[i]) / 2 + c0[i]
                # formula (31)
                s_inv = (
                    1 / s0[i]
                    + 0.5 * k[i].T @ k[i]
                    + 0.5 * np.trace(L_inv @ J[i].T @ J[i])
                )
                s[i] = 1 / s_inv

            if "index_ARD" in kwargs:
                index_ARD = kwargs["index_ARD"]
                n_ARD_param = len(index_ARD)
                L0[index_ARD, index_ARD] = 1 / (
                    m[index_ARD] ** 2 + L_inv[index_ARD, index_ARD]
                )
                r = 2 / (m[index_ARD] ** 2 + np.diag(L_inv)[index_ARD])
                d = 0.5 * np.ones(n_ARD_param)

            logger.debug(f"current mean: {m}")
            logger.debug(f"current precision: {L}")

            # free energy caluclation, formula (23) slightly rearranged
            # to account for the loop over all noise groups
            f_new = -0.5 * ((m - m0).T @ L0 @ (m - m0) + np.trace(L_inv @ L0))
            (sign, logdet) = np.linalg.slogdet(L)
            f_new += 0.5 * sign * logdet

            for i in range(N):
                f_new += -s[i] * c[i] / s0[i] + (len(k[i]) / 2 + c0[i] - 1) * (
                    np.log(s[i]) + special.digamma(c[i])
                )
                f_new += -0.5 * (k[i].T @ k[i] + np.trace(L_inv @ J[i].T @ J[i]))
                f_new += -s[i] * np.log(c[i]) - special.gammaln(c[i])
                f_new += -c[i] + (len(k[i]) / 2 + c[i] - 1) * (
                    np.log(s[i]) + special.digamma(c[i])
                )
                if "index_ARD" in kwargs:
                    for j in range(n_ARD_param):
                        f_new += (
                            (d[j] - 2) * (np.log(s[i]) - special.digamma(d[j]))
                            - d[j] * (1 + np.log(r[j]))
                            - special.gammaln(d[j])
                        )
            logger.debug(f"Free energy of iteration {i_iter} is {f_new}")

            if self.stop_criteria([s, c, m, L], f_new, i_iter):
                break

        delta_f = self.f_old - f_new
        logger.debug(f"Stopping VB. Iterations:{i_iter}, free energy change {delta_f}.")

        self.result.njev = model_error.n_jac
        self.result.nfev = model_error.n
        self.result.nit = i_iter

        self.result.param = MVN(self.param_stored[2], self.param_stored[3])
        self.result.noise = []
        for s, c in zip(self.param_stored[0], self.param_stored[1]):
            self.result.noise.append(Gamma(s, c))

        return self.result

    def stop_criteria(self, prms, f_new, i_iter):
        self.n_trials += 1

        # parameter update
        self.result.free_energies.append(f_new)

        if f_new > self.f_stored:
            self.param_stored = [np.copy(x) for x in prms]
            self.f_stored = f_new

        if f_new > self.f_old:
            self.n_trials = 0

        # stop?
        if self.n_trials >= self.n_trials_max:
            self.result.message = "Stopping because free energy did not "
            self.result.message += f"increase within {self.n_trials_max} iterations."
            return True

        if i_iter >= self.iter_max:
            self.result.message = "Stopping because the maximum number of "
            self.result.message = "iterations is reached."
            return True

        if abs(f_new - self.f_old) <= self.tolerance:
            self.result.message = "Tolerance reached!"
            self.result.success = True
            return True

        # go on.
        self.f_old = f_new
        return False

import numpy as np
import scipy.stats
import scipy.special as special

from numpy.linalg import multi_dot as multi_dot

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
    def __init__(self, s=1, c=1, name="Gamma"):
        self.s = np.atleast_1d(s).astype(float)  # shape
        self.c = np.atleast_1d(c).astype(float)  # scale
        self.name = name

    @property
    def mean(self):
        return self.s * self.c

    def pdf(self, xs, i=0):
        return scipy.stats.gamma.pdf(xs, a=self.s[i], scale=self.c[i])

    def __str__(self):
        return f"{self.name} with \n └── mean: {pretty_array(self.mean, 9)}"

    @classmethod
    def FromSD(cls, sds, shape=1.0):
        if isinstance(shape, float):
            shape = np.ones_like(sds) * shape

        assert len(shape) == len(sds)

        scale = []
        for s, sd in zip(shape, sds):
            scale.append(1.0 / sd ** 2 / s)

        return cls(shape, scale)


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


def splitted_k_J(model_error, mean):
    """
    Evaluates the model_error (k) and the jacobian d/d_mean(k) and splits
    the contributions according to the sensor pattern.
    """
    k = model_error(mean)
    J = model_error.jacobian(mean)
    pattern = model_error.noise_pattern
    return split(k, pattern), split(J, pattern)


def check_pattern(pattern, N=None):
    """
    Checks, if all the indices from 0..N (excluding N) are contained
    in the noise pattern.
    """
    if pattern is None:
        return

    flat_pattern = [item for sublist in pattern for item in sublist]

    if N is None:
        N = max(flat_pattern) + 1
    else:
        if N != max(flat_pattern) + 1:
            error = f"The highest noise pattern index {max(flat_pattern)} "
            error += f"does not match the length of the model error (N)!"
            raise ValueError(error)

    for i in range(N):
        if i not in flat_pattern:
            raise ValueError(f"Index {i} not contained in pattern {pattern}.")

    assert len(flat_pattern) == N


def split(k, pattern):
    """
    Splits the vector `k` into len(pattern) subvectors, where each k_i contains 
    the indices of `pattern[i]`.
    """
    if pattern is None:
        return [k]

    ks = []
    for p in pattern:
        ks.append(k[p])

    return ks


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
            f0 = self._f(x)
            x[iParam] += 2 * dx
            f1 = self._f(x)
            x[iParam] -= dx

            if iParam == 0:
                # allocate jac
                jac = np.empty([len(f0), len(x)])

            jac[:, iParam] = (f1 - f0) / (2 * dx)

        return jac


class VerifiedModelError:
    """
    Forwards the model_error and tries to add additional information that may
    not be provided by the user, e.g. a noise_pattern or a numeric jacobian.
    """

    def __init__(self, model_error):
        self._f = CountedEval(model_error)

        try:
            self.noise_pattern = model_error.noise_pattern
            check_pattern(self.noise_pattern)
            self.n_noise = len(self.noise_pattern)

        except AttributeError:
            logger.debug("No ModelError.noise_pattern. Assuming single noise.")
            self.noise_pattern = None
            self.n_noise = 1

        try:
            self._jac = model_error.jacobian
        except AttributeError:
            logger.debug("No ModelError.jacobian. Using central differences")
            self._jac = Jacobian(self._f)

        self.n_jac = 0

    def __call__(self, parameters):
        return self._f(parameters)

    def jacobian(self, parameters):
        # somehow, per definition in the algorithm of Chappell2009, jacobian
        # refers to MINUS dk/dparameters...
        self.n_jac += 1
        J = -self._jac(parameters)

        return J


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
        k(parameter_means):
            difference of the forward model to the data

        jacobian(parameter_means) [optional]:
            total jacobian of the forward model w.r.t. the parameters

        noise_pattern [optional]:
            mapping of the sensor index to a sensor type
            E.g.: noise_pattern [[0,3], [1,2,4]] means that there are two 
            noise terms. The first noise term affects sensors 0 and 3, 
            the second one sensors 1,2 and 4.
            If you provide a noise pattern, each row of k must be present 
            in the pattern.

            If noise_pattern is not provided, a single noise for all the 
            sensors is assumed.

    param0:
        multivariate normal distributed parameter prior

    noise0:
        gamma distrubuted noise prior
        If noise0 is None, a noninformative gamma prior is chosen 
        automatically according to the noise pattern.
    
    tolerance:
        free energy change that causes the algorithm to stop

    iter_max:
        maximum number of iterations

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

        model_error = VerifiedModelError(model_error)

        if noise0 is None:
            noise0 = noninformative_gamma_prior(model_error.n_noise)

        if model_error.n_noise != len(noise0.s):
            error = f"Your noise pattern contains {model_error.n_noise} noise "
            error += f" terms, your noise prior contains {len(noise0.s)}."
            raise ValueError(error)

        # adapt notation
        s = np.copy(noise0.s)
        c = np.copy(noise0.c)
        m = np.copy(param0.mean)
        L = np.copy(param0.precision)

        s0 = np.copy(s)
        c0 = np.copy(c)
        m0 = np.copy(m)
        L0 = np.copy(L)

        self.param_stored = [np.copy(s), np.copy(c), np.copy(m), np.copy(L)]

        k_full, J_full = model_error(m0), model_error.jacobian(m0)
        check_pattern(model_error.noise_pattern, len(k_full))
        k = split(k_full, model_error.noise_pattern)
        J = split(J_full, model_error.noise_pattern)

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

            k, J = splitted_k_J(model_error, m)

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
                f_new += -s[i] * np.log(c[i])
                f_new += -c[i] + (len(k[i]) / 2 + c[i] - 1) * (
                    np.log(s[i]) + special.digamma(c[i])
                )

            logger.debug(f"Free energy of iteration {i_iter} is {f_new}")

            if self.stop_criteria([s, c, m, L], f_new, i_iter):
                break

        delta_f = self.f_old - f_new
        logger.debug(f"Stopping VB. Iterations:{i_iter}, free energy change {delta_f}.")

        self.result.njev = model_error.n_jac
        self.result.nfev = model_error._f.n
        self.result.nit = i_iter

        self.result.param = MVN(self.param_stored[2], self.param_stored[3])
        self.result.noise = Gamma(self.param_stored[0], self.param_stored[1])

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
            self.result.message = f"Stopping because free energy did not "
            self.result.message += "increase within {self.n_trials_max} iterations."
            return True

        if i_iter >= self.iter_max:
            self.result.message = f"Stopping because the maximum number of "
            self.result.message = "iterations is reached."
            return True

        if abs(f_new - self.f_old) <= self.tolerance:
            self.result.message = f"Tolerance reached!"
            self.result.success = True
            return True

        # go on.
        self.f_old = f_new
        return False


def noninformative_gamma_prior(n_priors):
    """
    Suggested by @ilma following
    https://projecteuclid.org/euclid.ejs/1320416981
    or
    https://math.stackexchange.com/questions/449234/vague-gamma-prior
    """
    s = np.full(n_priors, 1.0 / 3.0)
    c = np.full(n_priors, 0.0)
    return Gamma(s=s, c=c)

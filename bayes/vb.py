import copy
import numpy as np
import scipy.stats
import scipy.special as special
from .jacobian import delta_x
import neptune.new as neptune
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
    def __init__(self, shape=1.0, scale=1.0, name="Gamma"):
        self.scale = scale
        self.shape = shape
        self.name = name

    @property
    def mean(self):
        return self.scale * self.shape

    def pdf(self, xs):
        return scipy.stats.gamma.pdf(xs, a=self.shape, scale=self.scale)

    def __repr__(self):
        return f"{self.name} with \n └── mean: {self.mean, 9}"

    @classmethod
    def FromSD(cls, sd, shape=1.0):
        return cls(shape, (1.0 / sd ** 2) / shape)

    @classmethod
    def Noninformative(cls):
        """
        Suggested by @ilma following
        https://projecteuclid.org/euclid.ejs/1320416981
        or
        https://math.stackexchange.com/questions/449234/vague-gamma-prior
        """
        return cls(scale=1.0 / 3.0, shape=0.0)


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


class VariationalBayesInterface:
    def __call__(self, number_vector):
        """
        Returns a dict of type 
            {noise_key : model_error_vector}
        """
        raise NotImplementedError()

    def jacobian(self, number_vector):
        """
        Returns a dict of type 
            {noise_key : d_model_error_d_number_vector_matrix}

        By default, this is a numeric Jacobian calculated by central
        differences.
        """
        """
        Calculates the derivative of `vb_model_error` w.r.t `number_vector`

        vb_model_error:
            function that takes the single argument of type `number_vector` and
            returns a dict of type {key : numpy_vector of length N}
        number_vector:
            vector of numbers of length M
        returns:
            dict of type {key : numpy_matrix of shape NxM}
        """
        x = np.copy(number_vector)

        for iParam in range(len(x)):
            dx = delta_x(x[iParam])

            x[iParam] -= dx
            fs0 = self(x)
            x[iParam] += 2 * dx
            fs1 = self(x)
            x[iParam] = number_vector[iParam]

            if iParam == 0:
                # allocate jac
                jac = {}
                for key, f0 in fs0.items():
                    jac[key] = np.empty([len(f0), len(x)])

            for n in fs0:
                jac[n][:, iParam] = -(fs1[n] - fs0[n]) / (2 * dx)

        return jac


class VBModelErrorWrapper(VariationalBayesInterface):
    def __init__(self, model_error):
        """
        For simple cases with only a single noise group, we want the
        model error for variational bayes to just return a vector instead of
        {"some_dummy_noise_key":vector}.
        Still, to match the VariationalBayesInterface, we use this adapter.
        """
        self.model_error = model_error

    def __call__(self, number_vector):
        k = self.model_error(number_vector)
        if not isinstance(k, dict):
            return {"tmp_noise": k}
        else:
            return k



def variational_bayes_neptune(model_error, param0, noise0=None, **kwargs):
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
    return vb.run_neptune(model_error, param0, noise0, **kwargs)

def free_energy(model_error, param0, noise0=None, **kwargs):

    vb = VB()
    return vb.free_energy(model_error, param0, noise0, **kwargs)



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

class ParameterEvolution:
    """
    The ParameterList serves as an input to the user-defined models. It is
    basically a name:value-dict that allows the user to access the parameters
    by name instead of some vector index, which could read:

        def my_model(prm):
            return prm["slope"] * some_coordinates + prm["offset"]

    """

    def __init__(self):
        self.p = {}

    def define(self, name, value=None):
        self.p[name] = value

    def __getitem__(self, name):
        return self.p[name]

    def __contains__(self, name):
        return name in self.p

    def __setitem__(self, name, value):
        """
        Calling parameter_list["A"]=0. when there is no parameter "A" defined
        may hide some bugs in the user code. Thus, we force parameters to be
        defined via `self.define(name)` before accessing it.
        """
        if name not in self:
            raise Exception("Call .define to define new parameters.")
        self.define(name, value)


class VBResult:
    """
    Somehow inspired by 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """

    def __init__(self):
        self.param = None
        self.noise = {}
        self.success = False
        self.message = ""
        self.free_energies = []
        self.f_max = -np.inf
        self.nit = 0
        self.param_evolution = ParameterEvolution()

    def __str__(self):
        s = "### VB Result ###\n"

        for member, value in vars(self).items():
            v = str(value).replace("\n", "\n" + " " * 17)
            s += f"  {member+':':14s} {v}\n"

        return s

    def save_evolution(self, mean, precision, shapes, scales, i_iter):
        noise = {}
        for n in shapes:
            noise[n] = Gamma(shape=shapes[n], scale=scales[n])
        self.param_evolution.define('iter_{:02d}'.format(i_iter),
                                    {'param': MVN(mean, precision),
                                     'noise': noise})

    def try_update(self, f_new, mean, precision, shapes, scales):
        self.free_energies.append(f_new)
        if f_new > self.f_max:
            # update
            self.f_max = f_new
            self.param = MVN(mean, precision)

            for n in shapes:
                self.noise[n] = Gamma(shape=shapes[n], scale=scales[n])


class VB:
    def __init__(self, n_trials_max=10, iter_max=50, tolerance=0.1):
        self.f_old = -np.inf
        self.param_stored = None
        self.n_trials = 0
        self.n_trials_max = n_trials_max
        self.tolerance = tolerance
        self.iter_max = iter_max
        self.result = VBResult()
    # @profile
    def run(self, model_error, param0, noise0=None, fix_noise=None, **kwargs):

        if "tolerance" in kwargs:
            self.tolerance = kwargs["tolerance"]
        if "iter_max" in kwargs:
            self.iter_max = kwargs["iter_max"]
        if "n_trials_max" in kwargs:
            self.n_trials_max = kwargs["n_trials_max"]

        if not isinstance(model_error, VariationalBayesInterface):
            model_error = VBModelErrorWrapper(model_error)

        k, J = model_error(param0.mean), model_error.jacobian(param0.mean)

        return_single_noise = False

        if noise0 is None:
            noise0 = {noise_key: Gamma.Noninformative() for noise_key in k}
            if len(noise0) == 1:
                return_single_noise = True

        update_noise = {}
        for i in noise0:
            if fix_noise == None:
                update_noise[i] = True
            elif i not in fix_noise:
                update_noise[i] = True
            else:
                update_noise[i] = False

        if isinstance(noise0, Gamma):
            # if a single Gamma is provided as prior, a single noise should
            # be returned as posterior.
            return_single_noise = True
            if len(k) != 1:
                error = "Passing a single Gamma distribution, so without the "
                error += "dict-pattern {noise_key : Gamma}, is only valid if "
                error += "the provided model error has a single noise group!"
                raise ValueError(error)
            noise_key = list(k.keys())[0]
            noise0 = {noise_key: noise0}

        for noise_key in k:
            if noise_key not in noise0:
                error = f"Your model error contains the noise key {noise_key},"
                error += f"which is not given in your noise prior!"
                raise ValueError(error)

        # adapt notation
        s, c = {}, {}
        for n, gamma in noise0.items():
            s[n] = gamma.scale
            c[n] = gamma.shape
        m = np.copy(param0.mean)
        L = np.copy(param0.precision)

        m0 = np.copy(m)
        L0 = np.copy(L)
        s0 = copy.copy(s)
        c0 = copy.copy(c)

        self.param_stored = [np.copy(s), np.copy(c), copy.copy(m), copy.copy(L)]

        i_iter = 0
        self.result.save_evolution(m, L, c, s, i_iter)
        while True:
            i_iter += 1

            # fw model parameter update
            L = sum([s[i] * c[i] * J[i].T @ J[i] for i in noise0]) + L0
            L_inv = np.linalg.inv(L)

            Lm = sum([s[i] * c[i] * J[i].T @ (k[i] + J[i] @ m) for i in noise0])
            Lm += L0 @ m0
            m = Lm @ L_inv

            k, J = model_error(m), model_error.jacobian(m)

            # noise parameter update
            for i in noise0:
                if update_noise[i]:
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
            f_new += - 0.5 * sign * logdet

            for i in noise0:
                f_new += -s[i] * c[i] / s0[i] + (len(k[i]) / 2 + c0[i] - 1) * (
                        np.log(s[i]) + special.digamma(c[i])
                )
                f_new += -0.5 * s[i] * c[i] * (k[i].T @ k[i] + np.trace(L_inv @ J[i].T @ J[i]))
                f_new += c[i] * np.log(s[i]) + special.gammaln(c[i])
                f_new += c[i] - (c[i] - 1) * (
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

            self.result.save_evolution(m, L, c, s, i_iter)
            self.result.try_update(f_new, m, L, c, s)
            if self.stop_criteria(f_new, i_iter):
                break

        delta_f = self.f_old - f_new
        logger.debug(f"Stopping VB. Iterations:{i_iter}, free energy change {delta_f}.")

        # self.result.njev = model_error.n_jac
        # self.result.nfev = model_error.n
        self.result.nit = i_iter

        if return_single_noise:
            assert len(self.result.noise) == 1
            noise = list(self.result.noise.values())[0]
            self.result.noise = noise

        return self.result

    def free_energy(self, model_error, param0, noise0=None, **kwargs):

        if not isinstance(model_error, VariationalBayesInterface):
            model_error = VBModelErrorWrapper(model_error)

        k, J = model_error(param0.mean), model_error.jacobian(param0.mean)

        return_single_noise = False

        if noise0 is None:
            noise0 = {noise_key: Gamma.Noninformative() for noise_key in k}
            if len(noise0) == 1:
                return_single_noise = True

        if isinstance(noise0, Gamma):
            # if a single Gamma is provided as prior, a single noise should
            # be returned as posterior.
            return_single_noise = True
            if len(k) != 1:
                error = "Passing a single Gamma distribution, so without the "
                error += "dict-pattern {noise_key : Gamma}, is only valid if "
                error += "the provided model error has a single noise group!"
                raise ValueError(error)
            noise_key = list(k.keys())[0]
            noise0 = {noise_key: noise0}

        for noise_key in k:
            if noise_key not in noise0:
                error = f"Your model error contains the noise key {noise_key},"
                error += f"which is not given in your noise prior!"
                raise ValueError(error)

        # adapt notation
        s, c = {}, {}
        for n, gamma in noise0.items():
            s[n] = gamma.scale
            c[n] = gamma.shape
        m = np.copy(param0.mean)
        L = np.copy(param0.precision)

        m0 = np.copy(m)
        L0 = np.copy(L)
        s0 = copy.copy(s)
        c0 = copy.copy(c)

        # fw model parameter update
        L_inv = np.linalg.inv(L)

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
        f_new += -0.5 * sign * logdet

        for i in noise0:
            f_new += -s[i] * c[i] / s0[i] + (len(k[i]) / 2 + c0[i] - 1) * (np.log(s[i]) + special.digamma(c[i]))
            f_new += -0.5* s[i] * c[i] * (k[i].T @ k[i] + np.trace(L_inv @ J[i].T @ J[i]))
            f_new += c[i] * np.log(s[i]) + special.gammaln(c[i])
            f_new += c[i] - (c[i] - 1) * (
                    np.log(s[i]) + special.digamma(c[i])
            )
            if "index_ARD" in kwargs:
                for j in range(n_ARD_param):
                    f_new += (
                            (d[j] - 2) * (np.log(s[i]) - special.digamma(d[j]))
                            - d[j] * (1 + np.log(r[j]))
                            - special.gammaln(d[j])
                    )
        return f_new

    def run_neptune(self, model_error, param0, noise0=None, **kwargs):
        run = kwargs["run"]

        if "tolerance" in kwargs:
            self.tolerance = kwargs["tolerance"]
        if "iter_max" in kwargs:
            self.iter_max = kwargs["iter_max"]
        if "n_trials_max" in kwargs:
            self.n_trials_max = kwargs["n_trials_max"]

        if not isinstance(model_error, VariationalBayesInterface):
            model_error = VBModelErrorWrapper(model_error)

        k, J = model_error(param0.mean), model_error.jacobian(param0.mean)

        return_single_noise = False

        if noise0 is None:
            noise0 = {noise_key: Gamma.Noninformative() for noise_key in k}
            if len(noise0) == 1:
                return_single_noise = True

        if isinstance(noise0, Gamma):
            # if a single Gamma is provided as prior, a single noise should
            # be returned as posterior.
            return_single_noise = True
            if len(k) != 1:
                error = "Passing a single Gamma distribution, so without the "
                error += "dict-pattern {noise_key : Gamma}, is only valid if "
                error += "the provided model error has a single noise group!"
                raise ValueError(error)
            noise_key = list(k.keys())[0]
            noise0 = {noise_key: noise0}

        for noise_key in k:
            if noise_key not in noise0:
                error = f"Your model error contains the noise key {noise_key},"
                error += f"which is not given in your noise prior!"
                raise ValueError(error)

        # adapt notation
        s, c = {}, {}
        for n, gamma in noise0.items():
            s[n] = gamma.scale
            c[n] = gamma.shape
        m = np.copy(param0.mean)
        L = np.copy(param0.precision)

        m0 = np.copy(m)
        L0 = np.copy(L)
        s0 = copy.copy(s)
        c0 = copy.copy(c)

        self.param_stored = [np.copy(s), np.copy(c), copy.copy(m), copy.copy(L)]

        i_iter = 0
        self.result.save_evolution(m, L, c, s, i_iter)
        while True:
            i_iter += 1

            # fw model parameter update
            L = sum([s[i] * c[i] * J[i].T @ J[i] for i in noise0]) + L0
            L_inv = np.linalg.inv(L)

            Lm = sum([s[i] * c[i] * J[i].T @ (k[i] + J[i] @ m) for i in noise0])
            Lm += L0 @ m0
            m = Lm @ L_inv

            k, J = model_error(m), model_error.jacobian(m)

            # noise parameter update
            for i in noise0:
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
            # f_new = -0.5 * ((m - m0).T @ L0 @ (m - m0) + np.trace(L_inv @ L0))
            # (sign, logdet) = np.linalg.slogdet(L)
            # f_new += 0.5 * sign * logdet
            #
            # for i in noise0:
            #     f_new += -s[i] * c[i] / s0[i] + (len(k[i]) / 2 + c0[i] - 1) * (
            #         np.log(s[i]) + special.digamma(c[i])
            #     )
            #     f_new += -0.5 * (k[i].T @ k[i] + np.trace(L_inv @ J[i].T @ J[i]))
            #     f_new += -s[i] * np.log(c[i]) - special.gammaln(c[i])
            #     f_new += -c[i] + (len(k[i]) / 2 + c[i] - 1) * (
            #         np.log(s[i]) + special.digamma(c[i])
            #     )
            f_new = -0.5 * ((m - m0).T @ L0 @ (m - m0) + np.trace(L_inv @ L0))
            (sign, logdet) = np.linalg.slogdet(L)
            f_new += - 0.5 * sign * logdet

            for i in noise0:
                f_new += -s[i] * c[i] / s0[i] + (len(k[i]) / 2 + c0[i] - 1) * (
                    np.log(s[i]) + special.digamma(c[i])
                )
                f_new += -0.5 * (k[i].T @ k[i] + np.trace(L_inv @ J[i].T @ J[i]))*s[i]*c[i] #edited
                f_new += +c[i] * np.log(s[i]) + special.gammaln(c[i]) #edited t
                f_new += +c[i] - (c[i] - 1) * (
                    np.log(s[i]) + special.digamma(c[i])#edited
                )

                if "index_ARD" in kwargs:
                    for j in range(n_ARD_param):
                        f_new += (
                            (d[j] - 2) * (np.log(s[i]) - special.digamma(d[j]))
                            - d[j] * (1 + np.log(r[j]))
                            - special.gammaln(d[j])
                        )
            logger.debug(f"Free energy of iteration {i_iter} is {f_new}")
            run["metrics/free_energy_per_iter"].log(f_new)
            self.result.save_evolution(m, L, c, s, i_iter)
            self.result.try_update(f_new, m, L, c, s)
            if self.stop_criteria(f_new, i_iter):
                break

        delta_f = self.f_old - f_new
        logger.debug(f"Stopping VB. Iterations:{i_iter}, free energy change {delta_f}.")

        # self.result.njev = model_error.n_jac
        # self.result.nfev = model_error.n
        self.result.nit = i_iter

        if return_single_noise:
            assert len(self.result.noise) == 1
            noise = list(self.result.noise.values())[0]
            self.result.noise = noise

        return self.result
    def stop_criteria(self, f_new, i_iter):
        self.n_trials += 1

        if f_new > self.f_old:
            self.n_trials = 0

        # Update free energy here such that the "stop_criteria" is testable
        # individually:
        self.result.f_max = max(self.result.f_max, f_new)

        # stop?
        if self.n_trials >= self.n_trials_max:
            print("n trials")
            self.result.message = "Stopping because free energy did not "
            self.result.message += f"increase within {self.n_trials_max} iterations."
            return True

        if i_iter >= self.iter_max:
            print("iter max")
            self.result.message = "Stopping because the maximum number of "
            self.result.message = "iterations is reached."
            return True

        if abs(f_new - self.f_old) <= self.tolerance:
            print("tolerance")
            self.result.message = "Tolerance reached!"
            self.result.success = True
            return True

        # go on.
        self.f_old = f_new
        return False

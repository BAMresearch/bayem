import copy
import logging
from time import perf_counter

import numpy as np
import scipy.special as special

logger = logging.getLogger(__name__)

from .distributions import MVN, Gamma


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


        def delta_x(x0):
            dx = x0 * 1.0e-7 # approx x0 * sqrt(machine precision)
            if dx == 0:
                dx = 1.0e-7
            return dx


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
                jac[n][:, iParam] = (fs1[n] - fs0[n]) / (2 * dx)

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
        self.t = None

    def __str__(self):
        s = "### VB Result ###\n"

        for member, value in vars(self).items():
            v = str(value).replace("\n", "\n" + " " * 17)
            s += f"  {member+':':14s} {v}\n"

        return s

    def try_update(self, f_new, mean, precision, shapes, scales, parameter_names):
        self.free_energies.append(f_new)
        if f_new > self.f_max:
            # update
            self.f_max = f_new
            self.param = MVN(
                mean, precision, name="MVN posterior", parameter_names=parameter_names
            )

            for n in shapes:
                self.noise[n] = Gamma(shape=shapes[n], scale=scales[n])


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
            * NOTE: This is differs from the definition in the Chapell paper
                    where it is defined as MINUS d(forward_model)/d(parameters)
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


class VB:
    def __init__(self, n_trials_max=10, iter_max=50, tolerance=0.1):
        self.f_old = -np.inf
        self.param_stored = None
        self.n_trials = 0
        self.n_trials_max = n_trials_max
        self.tolerance = tolerance
        self.iter_max = iter_max
        self.scale_by_prior_mean = True
        self.result = VBResult()
        self.scaling_eps = 1.0e-20

    def run(self, model_error, param0, noise0=None, update_noise=True, **kwargs):
        t0 = perf_counter()
        if "tolerance" in kwargs:
            self.tolerance = kwargs["tolerance"]
        if "iter_max" in kwargs:
            self.iter_max = kwargs["iter_max"]
        if "n_trials_max" in kwargs:
            self.n_trials_max = kwargs["n_trials_max"]
        if "scale_by_prior_mean" in kwargs:
            self.scale_by_prior_mean = kwargs["scale_by_prior_mean"]
        if "scaling_eps" in kwargs:
            self.scaling_eps = kwargs["scaling_eps"]

        if not isinstance(model_error, VariationalBayesInterface):
            model_error = VBModelErrorWrapper(model_error)

        # We perform a scaling of the prior to deal with numerically high
        # high values.
        scaling = np.ones_like(param0.mean)

        if self.scale_by_prior_mean:
            for i, mean in enumerate(param0.mean):
                if abs(mean) > self.scaling_eps:
                    scaling[i] = mean

        logger.debug(f"Using scaling {scaling}")

        P = np.diag(scaling)
        Pinv = np.diag(1.0 / scaling)

        k, J_orig = model_error(param0.mean), model_error.jacobian(param0.mean)
        J = {}
        for n, jac in J_orig.items():
            J[n] = jac @ P

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

        if isinstance(update_noise, bool):
            # apply the `update_noise` flag to all noises
            flag = update_noise
            update_noise = {}
            for i in noise0:
                update_noise[i] = flag

        # adapt notation
        s, c = {}, {}
        for n, gamma in noise0.items():
            s[n] = gamma.scale
            c[n] = gamma.shape
        m = Pinv @ param0.mean
        L = P @ param0.precision @ P

        m0 = np.copy(m)
        L0 = np.copy(L)
        s0 = copy.copy(s)
        c0 = copy.copy(c)

        self.param_stored = [np.copy(s), np.copy(c), copy.copy(m), copy.copy(L)]

        i_iter = 0
        while True:
            i_iter += 1

            # fw model parameter update
            L = sum([s[i] * c[i] * J[i].T @ J[i] for i in noise0]) + L0
            L_inv = np.linalg.inv(L)

            Lm = sum([s[i] * c[i] * J[i].T @ (-k[i] + J[i] @ m) for i in noise0])
            Lm += L0 @ m0
            m = Lm @ L_inv

            k, J_orig = model_error(P @ m), model_error.jacobian(P @ m)
            J = {}
            for n, jac in J_orig.items():
                J[n] = jac @ P

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

            Pm = P @ m
            PinvLPinv = Pinv @ L @ Pinv
            logger.debug(f"current mean: {Pm}")
            logger.debug(f"current precision: {PinvLPinv}")
            logger.debug(f"scaled current mean: {m}")
            logger.debug(f"scaled current precision: {L}")

            # free energy caluclation, formula (23) slightly rearranged
            # to account for the loop over all noise groups
            f_new = -0.5 * ((m - m0).T @ L0 @ (m - m0) + np.trace(L_inv @ L0))
            (sign, logdet) = np.linalg.slogdet(L)
            f_new -= 0.5 * sign * logdet

            for i in noise0:
                """
                NOTE:
                    Still a possible slight improve is to exclude constant terms
                    (depending only on c and s) in case of update_noise[i]=False.
                    This however seems to have no influence, since they are just constants.
                """
                f_new += -s[i] * c[i] / s0[i] + (len(k[i]) / 2 + c0[i] - 1) * (
                    np.log(s[i]) + special.digamma(c[i])
                )
                f_new += (
                    -0.5
                    * s[i]
                    * c[i]
                    * (k[i].T @ k[i] + np.trace(L_inv @ J[i].T @ J[i]))
                )
                f_new += c[i] * np.log(s[i]) + special.gammaln(c[i])
                f_new += c[i] - (c[i] - 1) * (np.log(s[i]) + special.digamma(c[i]))
                if "index_ARD" in kwargs:
                    for j in range(n_ARD_param):
                        f_new += (
                            (d[j] - 2) * (np.log(s[i]) - special.digamma(d[j]))
                            - d[j] * (1 + np.log(r[j]))
                            - special.gammaln(d[j])
                        )
            logger.debug(f"Free energy of iteration {i_iter} is {f_new}")

            self.result.try_update(f_new, Pm, PinvLPinv, c, s, param0.parameter_names)
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

        self.result.t = perf_counter() - t0
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



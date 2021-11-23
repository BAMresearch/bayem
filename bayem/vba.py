import copy
import json
import logging
from dataclasses import dataclass
from time import perf_counter
from tabulate import tabulate
from typing import Union, Dict, Tuple

import numpy as np
import scipy.special as special

from .distributions import MVN, Gamma

logger = logging.getLogger(__name__)


@dataclass
class Options:
    tolerance: float = 0.1
    maxiter: int = 50
    maxtrials: int = 10
    update_noise: Union[Dict, bool] = True
    index_ARD: Tuple[int] = ()


class CDF_Jacobian:
    def __init__(self, f, transformation):
        self._f = f
        self._T = transformation

    def __call__(self, _x):
        x = np.copy(_x)

        for iParam in range(len(x)):
            eps = 1.0e-7  # approx sqrt(machine precision)
            dx = max(eps, abs(x[iParam]) * eps)

            x[iParam] -= dx
            fs0 = self._T(self._f(x))
            x[iParam] += 2 * dx
            fs1 = self._T(self._f(x))
            x[iParam] = _x[iParam]

            if iParam == 0:
                # allocate jac
                jac = {}
                for key, f0 in fs0.items():
                    jac[key] = np.empty([len(f0), len(x)])

            for n in fs0:
                jac[n][:, iParam] = (fs1[n] - fs0[n]) / (2 * dx)

        return jac


class VBAProblem:
    def __init__(self, f, noise0, jac):
        self.f = f
        self.noise0 = noise0
        self.jac = jac
        self.jac_in_f = not callable(jac) and jac == True

        self._Tk = None  # transformation of k = f(x)
        self._TJ = None  # transformation of J = jac(x)

    def __call__(self, x):
        """
        Computes (f(x), jac(x)) at the current parameter mean `x` as a dict
        containing {noise_key: np.ndarray}.
        """
        if self.jac_in_f:
            k, J = self.f(x)
        else:
            k, J = self.f(x), self.jac(x)
        return self._Tk(k), self._TJ(J)

    def original_noise(self, n):
        return self._invTnoise(n)

    def first_call(self, x):
        """
        Computes (f(x), jac(x)) at the current parameter mean `x` and sets
        up an internal structure such that various user provided forms of
        `f` can be used.
        """

        def no_transformation(x_dict):
            return x_dict

        def list_to_dict(x_list):
            return dict(enumerate(x_list))

        def numpy_to_dict(x_np):
            return {0: x_np}

        def dict_to_gamma(x_dict):
            assert len(x_dict) == 1
            assert 0 in x_dict
            return x_dict[0]

        def dict_to_gamma_list(x_dict):
            return [x_dict[i] for i in range(len(x_dict))]

        if self.jac_in_f:
            k, J = self.f(x)
        else:
            k = self.f(x)

        if isinstance(k, dict):
            self._Tk = no_transformation
            self._TJ = no_transformation
            if self.noise0 is None:
                self.noise0 = {noise_group: Gamma() for noise_group in k}
            self._invTnoise = no_transformation

        elif isinstance(k, list):
            self._Tk = list_to_dict
            self._TJ = list_to_dict
            if self.noise0 is None:
                self.noise0 = [Gamma() for _ in range(len(k))]
            self._invTnoise = dict_to_gamma_list

        else:
            self._Tk = numpy_to_dict
            self._TJ = numpy_to_dict
            if self.noise0 is None:
                self.noise0 = Gamma()
            self._invTnoise = dict_to_gamma

        self.noise0 = self._Tk(self.noise0)

        if self.jac_in_f:
            pass
        else:
            if not self.jac:
                self.jac = CDF_Jacobian(self.f, self._Tk)
                self._TJ = no_transformation

            J = self.jac(x)

        return self._Tk(k), self._TJ(J)


def vba(f, x0, noise0=None, jac=None, **option_kwargs):
    """
    Implementation of
        Variational Bayesian inference for a nonlinear
        forward model [Chapell et al, 2008]
    for an arbitrary `model_error` f.

    Parameters
    ==========

    The implementation is close to the formulas of
    section III.C (Extending the Noise Model)
    with the same notation and references to each formula, with the only
    exception that capital lambda (precision) in the paper is here referred
    to as L.

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
    """

    vba_problem = VBAProblem(f, noise0, jac)
    return VBA(vba_problem, x0, Options(**option_kwargs)).run()


class VBA:
    def __init__(self, vba_problem, x0, options):
        self.p = vba_problem
        self.x0 = x0
        self.options = options
        self.result = VBResult()

        self.n_trials = 0
        self.f_old = -np.inf

    def run(self):
        t0 = perf_counter()
        # extract / rename model parameters
        m0 = self.x0.mean
        L0 = self.x0.precision
        self.m = np.array(self.x0.mean)
        self.L = np.array(self.x0.precision)

        # run first evaluation of the f and jac to adjust the format
        k, J = self.p.first_call(self.m)

        self.noise0 = self.p.noise0

        # extract / rename noise parameters
        self.s, self.c = {}, {}
        for n, gamma in self.noise0.items():
            self.s[n] = gamma.scale
            self.c[n] = gamma.shape
        s0 = copy.copy(self.s)
        c0 = copy.copy(self.c)

        self.noise_groups = self.noise0.keys()

        i_iter = 0
        while True:
            i_iter += 1

            self.update_parameters(k, J)

            k, J = self.p(self.m)

            self.update_noise(k, J)

            
            index_ARD = list(self.options.index_ARD)
            n_ARD_param = len(index_ARD)
            self.x0.precision[index_ARD, index_ARD] = 1 / (
                    self.m[index_ARD] ** 2 + self.L_inv[index_ARD, index_ARD]
                )
            r = 2 / (self.m[index_ARD] ** 2 + np.diag(self.L_inv)[index_ARD])
            d = 0.5 * np.ones(n_ARD_param)

            logger.info(f"current mean: {self.m}")

            f_new = self.free_energy(k, J)

            for i in self.noise_groups:
                for j in range(n_ARD_param):
                    f_new += (
                        (d[j] - 2) * (np.log(self.s[i]) - special.digamma(d[j]))
                        - d[j] * (1 + np.log(r[j]))
                        - special.gammaln(d[j])
                    )
            logger.info(f"Free energy of iteration {i_iter} is {f_new}")

            self.result.try_update(
                f_new, self.m, self.L, self.c, self.s, self.x0.parameter_names
            )
            if self.stop_criteria(f_new, i_iter):
                break

        delta_f = self.f_old - f_new
        logger.debug(f"Stopping VB. Iterations:{i_iter}, free energy change {delta_f}.")

        self.result.nit = i_iter

        self.result.t = perf_counter() - t0
        self.result.param0 = self.x0
        self.result.noise0 = self.p.original_noise(self.noise0)
        self.result.noise = self.p.original_noise(self.result.noise)
        return self.result

    def update_parameters(self, k, J):
        # fw model parameter update
        m0, L0 = self.x0.mean, self.x0.precision
        self.L = (
            sum([self.s[i] * self.c[i] * J[i].T @ J[i] for i in self.noise_groups]) + L0
        )
        self.L_inv = np.linalg.inv(self.L)

        Lm = sum(
            [
                self.s[i] * self.c[i] * J[i].T @ (-k[i] + J[i] @ self.m)
                for i in self.noise_groups
            ]
        )
        Lm += L0 @ m0
        self.m = Lm @ self.L_inv

    def update_noise(self, k, J):
        # noise parameter update
        for i in self.noise_groups:
            try:
                update_noise = self.options.update_noise[i]
            except TypeError:
                update_noise = self.options.update_noise

            if not update_noise:
                return


            # if update_noise[i]:
            # formula (30)
            c0i, s0i = self.noise0[i].shape, self.noise0[i].scale
            self.c[i] = len(k[i]) / 2 + c0i
            # formula (31)
            s_inv = (
                1 / s0i
                + 0.5 * k[i].T @ k[i]
                + 0.5 * np.trace(self.L_inv @ J[i].T @ J[i])
            )
            self.s[i] = 1 / s_inv

    def free_energy(self, k, J):
        m0, L0 = self.x0.mean, self.x0.precision

        # free energy caluclation, formula (23) slightly rearranged
        # to account for the loop over all noise groups
        f_new = -0.5 * (
            (self.m - m0).T @ L0 @ (self.m - m0) + np.trace(self.L_inv @ L0)
        )
        (sign, logdet) = np.linalg.slogdet(self.L)
        f_new -= 0.5 * sign * logdet
        f_new += 0.5 * len(self.m)

        (sign0, logdet0) = np.linalg.slogdet(L0)
        f_new += 0.5 * sign0 * logdet0

        for i in self.noise_groups:
            c0i, s0i = self.noise0[i].shape, self.noise0[i].scale
            si, ci = self.s[i], self.c[i]
            N = len(k[i])

            # From the update equation
            f_new += -si * ci / s0i + (N / 2 + c0i - 1) * (
                np.log(si) + special.digamma(ci)
            )
            f_new += (
                -0.5 * si * ci * (k[i].T @ k[i] + np.trace(self.L_inv @ J[i].T @ J[i]))
            )
            f_new += ci * np.log(si) + special.gammaln(ci)
            f_new += ci - (ci - 1) * (np.log(si) + special.digamma(ci))
            # constant terms to fix the evidence
            f_new += (
                -N / 2 * np.log(2 * np.pi) - special.gammaln(c0i) - c0i * np.log(s0i)
            )
           
        return f_new

    def stop_criteria(self, f_new, i_iter):
        self.n_trials += 1

        if f_new > self.f_old:
            self.n_trials = 0

        # Update free energy here such that the "stop_criteria" is testable
        # individually:
        self.result.f_max = max(self.result.f_max, f_new)

        # stop?
        if self.n_trials >= self.options.maxtrials:
            self.result.message = "Stopping because free energy did not "
            self.result.message += f"increase within {self.options.maxtrials} iterations."
            return True

        if i_iter >= self.options.maxiter:
            self.result.message = "Stopping because the maximum number of "
            self.result.message = "iterations is reached."
            return True

        if abs(f_new - self.f_old) <= self.options.tolerance:
            self.result.message = "Tolerance reached!"
            self.result.success = True
            return True

        # go on.
        self.f_old = f_new
        return False


    # """
    # Nonlinear variational bayes algorithm according to
    # @article{chappell2008variational,
    #       title={Variational Bayesian inference for a nonlinear forward model},
    #       author={Chappell, Michael A and Groves, Adrian R and Whitcher,
    #               Brandon and Woolrich, Mark W},
    #       journal={IEEE Transactions on Signal Processing},
    #       volume={57},
    #       number={1},
    #       pages={223--236},
    #       year={2008},
    #       publisher={IEEE}
    #     }
    #
    # This implements the formulas of section III.C (Extending the Noise Model)
    # with the same notation and references to each formula, with the only
    # exception that capital lambda L in the paper is here referred to as L.
    #
    # model_error that contains
    #     __call__(parameter_means):
    #         * difference of the forward model to the data
    #         * list of numpy vectors where each list corresponds to one noise group
    #         * alternatively: just a numpy vector for the case of exactly one
    #                          noise group
    #
    #     jacobian(parameter_means) [optional]:
    #         * total jacobian of the forward model w.r.t. the parameters
    #         * NOTE: This is differs from the definition in the Chapell paper
    #                 where it is defined as MINUS d(forward_model)/d(parameters)
    #         * list of numpy matrices where each list corresponds to one noise group
    #         * alternatively: just a numpy matrix for the case of exactly one
    #                          noise group
    # param0:
    #     multivariate normal distributed parameter prior
    #
    # noise0:
    #     list of gamma distributions for the noise prior
    #     If noise0 is None, a noninformative gamma prior is chosen
    #     automatically according to the number of noise groups.
    #
    # tolerance:
    #     free energy change that causes the algorithm to stop
    #
    # iter_max:
    #     maximum number of iterations
    #
    # index_ARD:
    #     Automatic Relevance Determination option to allow for "the automated reduction of model
    #     complexity" (Chappell et al. 2009). Should be passed (as kwargs) when ARD is applied to
    #     one of the model parameters described by the MVN. It should be an array containing
    #     the indexes corresponding to the position of the ARD parameters in the m and dig(L)
    #     vectors.
    #
    # Returns:
    #     VBResult defined below
    # """

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
        self.means = []
        self.sds = []
        self.shapes = []
        self.scales = []
        self.f_max = -np.inf
        self.nit = 0
        self.t = None

    def __str__(self):
        s = "### VB Result ###\n"

        for member in [
            "success",
            "message",
            "param",
            "noise",
            "free_energies",
            "f_max",
            "nit",
            "t",
        ]:
            to_print = self.__dict__[member]
            v = str(to_print).replace("\n", "\n" + " " * 17)
            s += f"  {member+':':14s} {v}\n"

        return s

    def try_update(self, f_new, mean, precision, shapes, scales, parameter_names):
        self.free_energies.append(f_new)
        self.means.append(mean)
        self.sds.append(MVN(mean, precision).std_diag)
        self.shapes.append(shapes)
        self.scales.append(scales)
        if f_new > self.f_max:
            # update
            self.f_max = f_new
            self.param = MVN(
                mean, precision, name="MVN posterior", parameter_names=parameter_names
            )

            for n in shapes:
                self.noise[n] = Gamma(shape=shapes[n], scale=scales[n])

    def summary(
        self,
        gamma_as_sd=False,
        printer=None,
        quantiles=[0.05, 0.25, 0.75, 0.95],
        **tabulate_kwargs,
    ):
        if printer is None:
            printer = print

        data = []
        p = self.param
        for i in range(len(p)):
            dist = p.dist(i)
            entry = [p.parameter_names[i], dist.median(), dist.mean(), dist.std()]
            entry += [dist.ppf(q) for q in quantiles]
            data.append(entry)

        if isinstance(self.noise, Gamma):
            noises = {"noise": self.noise}
        else:
            noises = self.noise

        for name, p in noises.items():
            dist = p.dist()

            if gamma_as_sd:
                entry = [name, "?", 1 / dist.mean() ** 0.5, "?"]
                entry += [1 / dist.ppf(q) ** 0.5 for q in quantiles]
            else:
                entry = [name, dist.median(), dist.mean(), dist.std()]
                entry += [dist.ppf(q) for q in quantiles]
            data.append(entry)

        headers = ["name", "median", "mean", "sd"]
        headers += [f"{int(q*100)}%" for q in quantiles]

        s = tabulate(data, headers=headers, **tabulate_kwargs)
        printer(s)
        return data

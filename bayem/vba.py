import logging
from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter
from tabulate import tabulate
from typing import Union, Dict, Tuple

import numpy as np
import scipy.special as special

from .distributions import MVN, Gamma

logger = logging.getLogger(__name__)


def vba(f, x0, noise0=None, jac=None, **option_kwargs):
    """
    Implementation of
        Variational Bayesian inference for a nonlinear
        forward model [Chapell et al, 2008]
    for an arbitrary `model_error` f.

    Parameters
    ==========
    f:
        model error callable that takes a parameter vector as input and
        returns one of:
            * numeric vector
            * list of numeric vectors where each item belongs to a separate
              "noise group"
            * dict of numeric vectors where each value belongs to a separate
              "noise group"

    x0:
        bayem.MVN multivariate normal distribution for the model parameter 
        prior

    noise0:
        Collection* (see below) bayem.Gamma distributions for the noise (hyper) 
        parameter prior. 
        If noise0 is None, a noninformative gamma prior is chosen
        automatically according to the number of noise groups.

    jac:
        callable that takes a parameter vector as input and returns a 
        collection* (see below) of df/dmodel_paramaters matrices.
        * jac == True indicates that the model error `f` returns a tuple 
          containing both the model error and its derivative.
        * jac == None/False falls back to a numeric implementation based on
          central differences of `f`. 

    option_kwargs:
        
        tolerance:
            The algorithm stops, if the change in the variational free energy 
            to the previous iteration is below `tolerance`.

        maxiter:
            The algorithm stops after `maxiter` iterations.

        maxtrails:
            The algorithm stops, if the free energy does not increase
            for `maxtrails` iterations.

        update_noise:
            Flags indicating whether or not the noise parameters should be 
            inferred. This can be passed as a single boolean for all noise
            groups or as a dict {noise_group_key : bool}.
        
        index_ARD:
            Automatic Relevance Determination option to allow for "the automated 
            reduction of model complexity" (Chappell et al. 2009). It should be 
            an array containing the indexes corresponding to the position of 
            the ARD parameters in the `x0` MVN.

        cdf_eps:
            epsilon for central differences jacobian, approx 
            sqrt(machine precision)

        store_full_precision:
            If true, includes the full precision for each iteration in the 
            VBResult. Set that to False for _big_ problems to save memory.


    Returns:
        bayem.VBResult defined below

    Additional notes:
    =================

    collection*:
        The output of `f` must match the output of `jac` and the format of 
        `noise0`. For clarity:

        type(f(theta)) | type(jac(theta)) | type(noise0)
       ----------------+------------------+-------------- 
           vector      |      matrix      |     Gamma
        list[vector]   |    list[matrix]  |  list[Gamma]
        dict{g:vector} |   dict{g:matrix} | dict{g:Gamma}



    The implementation is close to the formulas of section III.C (Extending the 
    Noise Model) with the same notation and references to each formula, with the 
    only exception that capital lambda (precision) in the paper is here referred
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

    options = VBOptions(**option_kwargs)
    dict_f = DictModelError(f, noise0, jac, options)
    return VBA(dict_f, x0, options).run()


@dataclass
class VBOptions:
    """
    Options to control the behavior of the analytical variational Bayes alorithm

    As this class is only used internally, see the documentation in `bayem.vba`
    below
    """

    tolerance: float = 0.1
    maxiter: int = 50
    maxtrials: int = 10
    update_noise: Union[Dict, bool] = True
    index_ARD: Tuple[int] = ()

    cdf_eps: float = np.finfo(float).eps ** 0.5

    store_full_precision: bool = True


class CDF_Jacobian:
    def __init__(self, f, transformation, cdf_eps):
        """
        Provides a numerical jacobian df/dtheta for the user-provided callable
        `f` based on central differences.

        f:
            user provided callable, see `bayem.vba`

        transformation:
            callable that turns the output of `f` into the `dict` format
            of the `DictModelError` class. See the "additional notes" section
            in `bayem.vba`

        cdf_eps:
            defines the step length of parameter `x` as 
                h = max(cdf_eps, abs(x) * cdf_eps)
        """
        self._f = f
        self._T = transformation
        self.cdf_eps = cdf_eps

    def __call__(self, _x):
        x = np.copy(_x)

        for iParam in range(len(x)):
            eps = self.cdf_eps
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


def _identity(x):
    return x


def _list_to_dict(x_list):
    return dict(enumerate(x_list))


def _dict_to_list(x_dict):
    return [x_dict[i] for i in range(len(x_dict))]


def _obj_to_dict(x_np):
    return {0: x_np}


def _dict_to_obj(x_dict):
    assert len(x_dict) == 1
    assert 0 in x_dict
    return x_dict[0]


class DictModelError:
    """
    As indicated in `bayem.vba::AdditionalNotes`, the output of the user
    defined model error `f` may have various types. This class determines
    this type ["dict", "list", "other"] and applies the following 
    _transformations_ to transform the output to a dict structure. 
    """

    to_dict = {
        "dict": _identity,
        "list": _list_to_dict,
        "other":_obj_to_dict,
    }
    
    from_dict = {
        "dict": _identity,
        "list": _dict_to_list,
        "other": _dict_to_obj,
    }


    def __init__(self, f, noise0, jac, options):
        self.f = f
        self.noise0 = noise0
        self.jac = jac
        self.jac_in_f = not callable(jac) and jac
        self.options = options

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

        if self.jac_in_f:
            k, J = self.f(x)
        else:
            k = self.f(x)

        k_type = "other"
        default_noise = Gamma()
        if isinstance(k, dict):
            k_type = "dict"
            default_noise = {group: Gamma() for group in k}
        if isinstance(k, list):
            k_type = "list"
            default_noise = [Gamma() for _ in range(len(k))]

        self._Tk = self.to_dict[k_type]
        self._TJ = self.to_dict[k_type]
        self._invTnoise = self.from_dict[k_type]
        if self.noise0 is None:
            self.noise0 = default_noise

        self.noise0 = self._Tk(self.noise0)

        if self.jac_in_f:
            pass
        else:
            if not self.jac:
                self.jac = CDF_Jacobian(self.f, self._Tk, self.options.cdf_eps)
                self._TJ = _identity

            J = self.jac(x)

        return self._Tk(k), self._TJ(J)


class VBA:
    def __init__(self, dict_f, x0, options):
        self.p = dict_f
        self.x0 = x0
        self.options = options
        self.result = VBResult(options)

        self.n_trials = 0
        self.f_old = -np.inf

    def run(self):
        t0 = perf_counter()
        # extract / rename model parameters
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

        self.noise_groups = self.noise0.keys()

        i_iter = 0
        while True:
            i_iter += 1

            self.update_parameters(k, J)
            
            k, J = self.p(self.m)
            
            self.update_noise(k, J)

            for idx in self.options.index_ARD:
                mean = self.m[idx]
                var = 1 / self.L[idx, idx]
                new_var = mean ** 2 + var
                self.x0.precision[idx, idx] = 1 / new_var

            logger.info(f"current mean: {self.m}")

            f_new = self.free_energy(k, J)

            for i in self.options.index_ARD:
                d = 0.5
                r = 2 * self.x0.precision[i, i]
                f_new -= (d - 2) * special.digamma(d)
                f_new -= d * (1 + np.log(r))
                f_new -= special.gammaln(d)
                for noise in self.noise_groups:
                    f_new += (d - 2) * np.log(self.s[noise])

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
        self.result.noise0_dict = self.noise0
        self.result.noise_dict = self.result.noise
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
            c0i, s0i = self.noise0[i].shape, self.noise0[i].scale

            if update_noise:
                # formula (30)
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
            self.result.message += (
                f"increase within {self.options.maxtrials} iterations."
            )
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


class VBResult:
    """
    Somehow inspired by
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """

    def __init__(self, options):
        self.options = options
        self.param = None
        self.noise = {}
        self.success = False
        self.message = ""
        self.free_energies = []
        self.means = []
        self.sds = []
        self.gammas = defaultdict(list)
        self.precisions = []
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

        for n in shapes:
            self.gammas[n].append(Gamma(shape=shapes[n], scale=scales[n]))

        if self.options.store_full_precision:
            self.precisions.append(precision)
        if f_new > self.f_max:
            # update
            self.f_max = f_new
            self.param = MVN(mean, precision, "MVN posterior", parameter_names)

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


def _identity(x):
    return x


def _list_to_dict(x_list):
    return dict(enumerate(x_list))


def _dict_to_list(x_dict):
    return [x_dict[i] for i in range(len(x_dict))]


def _obj_to_dict(x_np):
    return {0: x_np}


def _dict_to_obj(x_dict):
    assert len(x_dict) == 1
    assert 0 in x_dict
    return x_dict[0]

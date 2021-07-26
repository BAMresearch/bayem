import numpy as np
from time import perf_counter
import collections

from .latent import LatentParameters
from .jacobian import jacobian_cdf
from .vb import MVN, Gamma, VariationalBayesInterface, variational_bayes


class SolverInterface:
    def __init__(self, problem):
        self.ts = []
        self.problem = problem

    def time_summary(self, printer=None):
        if printer is None:
            printer = print

        ts = self.ts
        # metaprogramming ftw
        s = ""
        for method in ["len", "np.sum", "np.mean", "np.median"]:
            s += f"{method}(ts)={eval(method)(ts):6.3e}s | "

        printer(s + "\n")


class VariationalBayesSolver(SolverInterface, VariationalBayesInterface):
    def __init__(self, *args, **kwargs):
        """
        The VariationalBayesSolver needs to separate model parameters
        and noise parameters. Thus, it needs to adapt the underlying
        inference problem such that the noise parameters are removed
        from the latent parameters.

        Additionally, we have the following restrictions:
            * The prior distribtions of the model parameters must be normals
            * There is exactly one noise precision parameter per noise model
            * The name of the noise precision parameter must match the
              noise model key.
        """
        super().__init__(*args, **kwargs)
        
        self._vb_latent = LatentParameters()
        for me_key, me in self.problem.model_errors.items():
            self._vb_latent.add_model(me_key, me)

        for global_name, latent in self.problem.latent.items():
            if global_name in self.problem.noise_models:
                continue
            self._vb_latent[global_name] = self.problem.latent[global_name]

    def __call__(self, number_vector, concatenate=True):
        """
        overwrites VariationalBayesInterface.__call__
        """
        t0 = perf_counter()
        updated_model_parameters = self._vb_latent.updated_parameters(number_vector)
        me = self.problem.evaluate_model_errors(updated_model_parameters)

        errors_by_noise = {}
        for key, noise in self.problem.noise_models.items():
            terms = noise.model_error_terms(me)
            if concatenate:
                terms = np.concatenate(terms)
            errors_by_noise[key] = terms

        self.ts.append(perf_counter() - t0)
        return errors_by_noise

    def jacobian(self, number_vector, concatenate=True):
        """
        overwrites VariationalBayesInterface.jacobian
        """
        updated_latent_parameters = self._vb_latent.updated_parameters(
            number_vector
        )
        jac = {}
        for me_key, me in self.problem.model_errors.items():
            me_parameter_list = updated_latent_parameters[me_key]

            if hasattr(me, "jacobian"):
                sensor_parameter_jac = me.jacobian(me_parameter_list)
            else:
                # The alternative would be a try/except block catching
                # AttributeError, but an attribute error may also occur
                # _within_ the jacobian evaluation. So we explicitly
                # check for the "jacobian" attribute.
                sensor_parameter_jac = jacobian_cdf(me, me_parameter_list)

            """
            sensor_parameter_jac contains a 
                dict (sensor) of 
                dict (parameter)
            
            We now "flatten" the last dict (parameter) in the order of the 
            latent parameters for a valid VB input. 

            """

            sensor_jac = {}
            for sensor, parameter_jac in sensor_parameter_jac.items():
                first_jac = list(parameter_jac.values())[0]
                N = len(first_jac)

                stacked_jac = np.zeros((N, len(number_vector)))

                for local_name in me_parameter_list.names:
                    global_name = self.problem.latent.global_name(me_key, local_name)
                    indices = self.problem.latent.global_indices(global_name)

                    J = parameter_jac[local_name]
                    # If it is a scalar parameter, the user may have
                    # defined as a vector of length N. We need to
                    # transform it to a matrix Nx1.
                    if len(J.shape) == 1:
                        J = np.atleast_2d(J).T

                    stacked_jac[:, indices] += J
                sensor_jac[sensor] = stacked_jac

            jac[me_key] = sensor_jac

        jacs_by_noise = {}
        for key, noise in self.problem.noise_models.items():
            terms = noise.jacobian_terms(jac)
            if concatenate:
                terms = np.concatenate(terms)
            jacs_by_noise[key] = terms

        return jacs_by_noise

    def prior_mvn(self):
        latent_prms = self._vb_latent

        latent_prms.check_priors()
        means = []
        precs = []

        for name, latent in latent_prms.items():
            assert latent.prior.dist.name == "norm"
            mean, sd = latent.prior.mean(), latent.prior.std()
            for _ in range(latent.N):
                means.append(mean)
                precs.append(1.0 / sd ** 2)

        return MVN(
            means,
            np.diag(precs),
            name="MVN prior",
            parameter_names=list(latent_prms.keys()),
        )

    def prior_noise(self):
        noise_prior = {}
        for noise_key, noise_model in self.problem.noise_models.items():
            latent = self.problem.latent[noise_key]
            assert len(latent) == 1
            assert latent[0][0] == noise_key
            assert latent[0][1] == "precision"
            assert latent.prior.dist.name == "gamma"

            mean, var = latent.prior.mean(), latent.prior.var()
            scale = var / mean
            shape = mean / scale
            noise_prior[noise_key] = Gamma(scale=scale, shape=shape)
        return noise_prior


    def estimate_parameters(self, **kwargs):
        param0 = self.prior_mvn()
        noise0 = self.prior_noise()
        return variational_bayes(self, param0, noise0, **kwargs)


class TaralliSolver(SolverInterface):
    def prior_transform(self, number_vector):
        ppf = []

        start_idx = 0
        for name, latent in self.problem.latent.items():
            theta = number_vector[start_idx : start_idx + latent.N]
            if len(theta) == 1:
                theta = theta[0]
            ppf.append(latent.prior.ppf(theta))
            start_idx += latent.N

        return np.array(ppf)

    def logprior(self, number_vector):
        """
        For taralli, number_vector has shape [n_processes x n_param] and must
        output a vector of length [n_processes] logpriors
        """
        number_vector2d = np.atleast_2d(number_vector)
        p = np.zeros(number_vector2d.shape[0])
        for i, serial_number_vector in enumerate(number_vector2d):
            start_idx = 0
            for name, latent in self.problem.latent.items():
                theta = number_vector[start_idx : start_idx + latent.N]
                if len(theta) == 1:
                    theta = theta[0]
                p[i] += latent.prior.logpdf(theta)
                start_idx += latent.N
        return p

    def loglike(self, number_vector):
        """
        For taralli, number_vector has shape [n_processes x n_param] and must
        output a vector of length [n_processes] loglikes
        """
        t0 = perf_counter()
        number_vector2d = np.atleast_2d(number_vector)
        ll = np.zeros(number_vector2d.shape[0])
        for i, serial_number_vector in enumerate(number_vector2d):
            try:
                ll[i] = self.problem.loglike(serial_number_vector)
            except ValueError as e:
                ll[i] = -np.inf
        self.ts.append(perf_counter() - t0)
        return ll

    def initial_samples(self, n):
        """
        Takes n samples from the prior and arranges them in a [n x n_prior] 
        matrix. 
        """
        init = np.empty((n, self._ndim()))
                
        for i, latent in enumerate(self.problem.latent.values()):
            init[:, i] = latent.prior.rvs(n)
        return init

    def _ndim(self):
        return self.problem.latent.vector_length

    def nestle_model(self, **kwargs):
        from taralli.parameter_estimation.base import NestleParameterEstimator

        return NestleParameterEstimator(
            ndim=self._ndim(),
            log_likelihood=self.loglike,
            prior_transform=self.prior_transform,
            **kwargs,
        )

    def dynesty_model(self, **kwargs):
        from taralli.parameter_estimation.base import DynestyParameterEstimator

        return DynestyParameterEstimator(
            ndim=self._ndim(),
            log_likelihood=self.loglike,
            prior_transform=self.prior_transform,
            **kwargs,
        )

    def emcee_model(self, nwalkers=20, **kwargs):
        from taralli.parameter_estimation.base import EmceeParameterEstimator

        init = self.initial_samples(nwalkers)
        return EmceeParameterEstimator(
            nwalkers=init.shape[0],
            ndim=init.shape[1],
            sampling_initial_positions=init,
            log_likelihood=self.loglike,
            log_prior=self.logprior,
            **kwargs,
        )

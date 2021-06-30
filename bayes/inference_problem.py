from collections import OrderedDict
import numpy as np
import scipy.stats
from .parameters import ParameterList
from .latent import LatentParameters
from .vb import MVN, Gamma, variational_bayes, VariationalBayesInterface
from .jacobian import d_model_error_d_named_parameter


class ModelErrorInterface:
    def __init__(self):
        self.parameter_list = ParameterList()

    def __call__(self):
        """
        Evaluate the model error based on `self.parameter_list` as a dict of
        {some_key: numpy.array}.
        """
        raise NotImplementedError("Override this!")

    def jacobian(self, latent_names=None):
        jac = dict()
        latent_names = latent_names or self.parameter_list.names
        for prm_name in latent_names:
            prm_jac = d_model_error_d_named_parameter(self, prm_name)
            for key in prm_jac:
                if key not in jac:
                    jac[key] = dict()

                jac[key][prm_name] = prm_jac[key]

        return jac


class InferenceProblem:
    def __init__(self):
        self.latent = LatentParameters()
        self.model_errors = OrderedDict()  # key : model_error
        self._noise_models = OrderedDict()  # key : noise_model
        self.prm_prior = {}
        self.noise_prior = {}

    @property
    def noise_models(self):
        if not self._noise_models:
            raise RuntimeError(
                "You need to define and add a noise model first! See `bayes.noise` for options and then call `.add_noise_model` to add it to the inference problem."
            )
        return self._noise_models

    def add_model_error(self, model_error, key=None):

        key = key or len(self.model_errors)
        assert key not in self.model_errors

        self.model_errors[key] = model_error
        return key

    def add_noise_model(self, noise_model, key=None):

        key = key or f"noise{len(self._noise_models)}"

        assert key not in self._noise_models
        self._noise_models[key] = noise_model
        return key

    def set_parameter_prior_normal(self, latent_name, mean, sd):
        """
        Sets a prior distribution for the latent parameter `latent_name` as
        a normal distribution with given `mean` and `sd`.
        """
        dist = scipy.stats.norm(loc=mean, scale=sd)
        self.set_parameter_prior(latent_name, dist)

    def set_parameter_prior(self, latent_name, dist):
        """
        Sets a prior distribution for the latent parameter `latent_name` to
        the provided `dist`. 
        """
        if latent_name not in self.latent:
            raise RuntimeError(
                f"{latent_name} is not defined as a latent parameter. "
                f"Call InferenceProblem.latent[{latent_name}].add(...) first."
            )

        if hasattr(dist, "ppf") and hasattr(dist, "logpdf"):
            self.prm_prior[latent_name] = dist
        else:
            raise RuntimeError(
                f"The provided distribution must provide a `ppf` and `logpdf` method (e.g. any scipy.stats distribution.)"
            )

    def set_noise_precision_prior(self, noise_key, dist):
        """
        Sets the prior distribution `dist` for the _precision_ of the 
        zero-mean noise term of `noise_key`. 
        """
        if noise_key not in self.noise_models:
            raise RuntimeError(
                f"{noise_key} is not associated with noise model.. "
                f"Call InferenceProblem.add_noise_model({noise_key}, ...) first."
            )
        if hasattr(dist, "ppf") and hasattr(dist, "logpdf"):
            self.noise_prior[noise_key] = dist
        else:
            raise RuntimeError(
                f"The provided distribution must provide a `ppf` and `logpdf` method (e.g. any scipy.stats distribution.)"
            )

    def set_noise_precision_prior_sd(self, noise_key, sd_mean, shape=1.0):
        """
        Sets a prior distribution for the _precision_ of the zero-mean noise 
        term of `noise_key` to a Gamma distribution with shape `shape` and
        mean 1/`sd_mean`**2.

        """
        a = shape
        scale = 1.0 / sd_mean ** 2 / shape
        dist = scipy.stats.gamma(a=shape, scale=scale)
        self.set_noise_precision_prior(noise_key, dist)

    def __call__(self, number_vector):
        self.latent.update(number_vector)
        result = {}
        for key, me in self.model_errors.items():
            result[key] = me()
        return result

    def define_shared_latent_parameter_by_name(self, name):
        for model_error in self.model_errors.values():
            try:
                prm = model_error.parameter_list
            except AttributeError:
                raise AttributeError(
                    "This method requires the `model_error` to have a `parameter_list` attribute!"
                )

            if name in model_error.parameter_list:
                self.latent[name].add(model_error.parameter_list, name)

    def loglike(self, number_vector):
        raw_me = self(number_vector)

        log_like = 0.0
        for noise_key, noise_term in self.noise_models.items():
            log_like += noise_term.loglike_contribution(raw_me)

        return log_like


class VariationalBayesProblem(InferenceProblem, VariationalBayesInterface):
    def run(self):
        info = variational_bayes(self, self.prior_MVN(), self.prior_gamma())
        return info

    def jacobian(self, number_vector, concatenate=True):
        """
        overwrites VariationalBayesInterface.jacobian
        """
        self.latent.update(number_vector)
        jac = {}
        for key, me in self.model_errors.items():

            # For each global latent parameter, we now need to find its
            # _local_ name, so the name in the parameter_list of the
            # model_error ...
            latent_names = self.latent.latent_names(me.parameter_list)
            local_latent_names = [n[0] for n in latent_names]
            # ... and only request the jacobian for the latent parameters.
            sensor_parameter_jac = me.jacobian(local_latent_names)
            """
            sensor_parameter_jac contains a 
                dict (sensor) of 
                dict (parameter)
            
            We now flatten the last dict (parameter) in the order of the 
            latent parameters for a valid VB input.

            This is challenging/ugly because:
                * The "parameter" in sensor_parameter_jac is not the same
                  as the corresponding _global_ parameter in the latent
                  parameters.
                * Some of the latent parameters may not be part of 
                  sensor_parameter_jac, because it only is a parameter of a 
                  different model error. We have to fill it with zeros of the
                  right dimension

            """
            sensor_jac = {}
            for sensor, parameter_jac in sensor_parameter_jac.items():
                first_jac = list(parameter_jac.values())[0]
                N = len(first_jac)

                # We allocate "stacked_jac" where each column corresponds
                # to a number in the "number_vector".
                stacked_jac = np.zeros((N, len(number_vector)))

                for (local_name, global_name) in latent_names:
                    J = parameter_jac[local_name]

                    # If it is a scalar parameter, the user may have
                    # defined as a vector of length N. We need to
                    # transform it to a matrix Nx1.
                    if len(J.shape) == 1:
                        J = np.atleast_2d(J).T

                    stacked_jac[:, self.latent[global_name].global_index_range()] += J

                sensor_jac[sensor] = stacked_jac

            jac[key] = sensor_jac

        jacs_by_noise = {}
        for key, noise in self.noise_models.items():
            terms = noise.jacobian_terms(jac)
            if concatenate:
                terms = np.concatenate(terms)
            jacs_by_noise[key] = terms

        return jacs_by_noise

    def __call__(self, number_vector, concatenate=True):
        """
        overwrites VariationalBayesInterface.__call__
        """
        me = super().__call__(number_vector)

        errors_by_noise = {}
        for key, noise in self.noise_models.items():
            terms = noise.model_error_terms(me)
            if concatenate:
                terms = np.concatenate(terms)
            errors_by_noise[key] = terms

        return errors_by_noise

    def prior_MVN(self):

        means = []
        precs = []

        for name, latent in self.latent.items():
            if name not in self.prm_prior:
                raise RuntimeError(
                    f"You defined {name} as latent but did not provide a prior distribution!."
                )

            if self.prm_prior[name].dist.name != "norm":
                raise RuntimeError(
                    f"VB problem can only handle normal priors, you provided `{self.prm_prior[name].dist.name}` for parameter `{name}`."
                )

            mean, var = self.prm_prior[name].mean(), self.prm_prior[name].var()
            for _ in range(latent.N):
                means.append(mean)
                precs.append(1.0 / var)

        return MVN(
            means,
            np.diag(precs),
            name="MVN prior",
            parameter_names=list(self.latent.keys()),
        )

    def prior_gamma(self):
        gammas = {}
        for name, gamma in self.noise_prior.items():
            mean, var = gamma.mean(), gamma.var()
            scale = var / mean
            shape = mean / scale
            gammas[name] = Gamma(scale=scale, shape=shape)
        return gammas

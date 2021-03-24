import numpy as np
from .parameters import ParameterList
from .latent import LatentParameters
from .noise import SingleSensorNoise
from collections import OrderedDict
from .vb import MVN, Gamma, variational_bayes, VariationalBayesModelError
from .jacobian import *


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

            try:
                N = len(self.parameter_list[prm_name])
                prm_jac = d_model_error_d_vector_parameter(self, prm_name)
            except TypeError:
                prm_jac = d_model_error_d_scalar_parameter(self, prm_name)

            for key in prm_jac:
                if key not in jac:
                    jac[key] = dict()

                jac[key][prm_name] = prm_jac[key]

        return jac


class InferenceProblem:
    def __init__(self):
        self.latent = LatentParameters()
        self.model_errors = OrderedDict()  # key : model_error
        self.noise_models = OrderedDict()  # key : noise_model

    def add_model_error(self, model_error, key=None):

        key = key or len(self.model_errors)
        assert key not in self.model_errors

        self.model_errors[key] = model_error
        return key

    def add_noise_model(self, noise_model, key=None):

        key = key or f"noise{len(self.noise_models)}"

        assert key not in self.noise_models
        self.noise_models[key] = noise_model
        return key

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
        self.latent.update(number_vector)
        raw_me = {}
        for key, me in self.model_errors.items():
            raw_me[key] = me()

        log_like = 0.0
        for noise_key, noise_term in self.noise_models.items():
            log_like += noise_term.loglike_contribution(raw_me)

        return log_like


class VariationalBayesProblem(InferenceProblem, VariationalBayesModelError):
    def __init__(self):
        super().__init__()
        self.prm_prior = {}
        self.noise_prior = {}

    def set_normal_prior(self, latent_name, mean, sd):
        if latent_name not in self.latent:
            raise RuntimeError(
                f"{latent_name} is not defined as a latent parameter. "
                f"Call InferenceProblem.latent[{latent_name}].add(...) first."
            )
        self.prm_prior[latent_name] = (mean, sd)

    def set_noise_prior(self, name, gamma_or_sd_mean, sd_scale=None):
        if isinstance(gamma_or_sd_mean, Gamma):
            gamma = gamma_or_sd_mean
            assert sd_scale is None
        else:
            sd_scale = sd_scale or 1.0
            gamma = Gamma.FromSD(gamma_or_sd_mean, sd_scale)

        if name not in self.noise_models:
            raise RuntimeError(
                f"{name} is not associated with noise model.. "
                f"Call InferenceProblem.add_noise_model({name}, ...) first."
            )
        self.noise_prior[name] = gamma

    def _use_default_noise(self):
        if not self.noise_models:
            default = SingleSensorNoise()
            noise_key = self.add_noise_model(default)
            self.noise_prior = Gamma.Noninformative()

    def run(self):
        self._use_default_noise()
        MVN = self.prior_MVN()
        noise = self.prior_noise()
        info = variational_bayes(self, MVN, noise)
        return info

    def jacobian(self, number_vector):
        self.latent.update(number_vector)
        jac = {}
        for key, me in self.model_errors.items():
            sensor_parameter_jac = me.jacobian()
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

                # For each global latent parameter, we now need to find its
                # _local_ name, so the name in the parameter_list of the
                # model_error.
                for latent in self.latent.values():
                    for (prm_list, name) in latent:
                        if prm_list == me.parameter_list:
                            # The _global_ latent parameter is part of the
                            # current model error!
                            J = parameter_jac[name]

                            # If it is a scalar parameter, the user may have
                            # defined as a vector of length N. We need to
                            # transform it to a matrix Nx1.
                            if len(J.shape) == 1:
                                J = np.atleast_2d(J).T

                            stacked_jac[:, latent.global_index_range()] = -J
                        else:
                            # Just for explaination. The global parameter is
                            # _not_ a parameter of the model_error and we
                            # need fill it with zeros. This is done by
                            # initializing "stacked_jac" with zeros above.
                            pass

                sensor_jac[sensor] = stacked_jac

            jac[key] = sensor_jac

        jacs_by_noise = {}
        for key, noise in self.noise_models.items():
            jacs_by_noise[key] = noise.jacobian_contribution(jac)

        return jacs_by_noise

    def __call__(self, number_vector):
        self._use_default_noise()
        me = super().__call__(number_vector)

        errors_by_noise = {}
        for key, noise in self.noise_models.items():
            errors_by_noise[key] = noise.vector_contribution(me)

        return errors_by_noise

    def prior_MVN(self):

        means = []
        precs = []

        for name, latent in self.latent.items():
            if name not in self.prm_prior:
                raise RuntimeError(
                    f"You defined {name} as latent but did not provide a prior distribution!."
                )
            mean, sd = self.prm_prior[name]
            for _ in range(latent.N):
                means.append(mean)
                precs.append(1.0 / sd ** 2)

        return MVN(means, np.diag(precs))

    def prior_noise(self):
        return self.noise_prior

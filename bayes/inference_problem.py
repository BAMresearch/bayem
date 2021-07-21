import numpy as np
from .parameters import ParameterList
from .latent import LatentParameters
from collections import OrderedDict
from .vb import MVN, Gamma, variational_bayes, VariationalBayesInterface
from .jacobian import d_model_error_d_named_parameter
from typing import Hashable


class ModelErrorInterface:
    def __call__(self, latent_parameter_list: ParameterList) -> dict[Hashable: np.ndarray]: 
        """
        Evaluate the model error based on the `latent_parameter_list`.
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

    def get_length(self, parameter_name):
        """
        Overwrite for more complex behaviours, e.g. to 

        ~~~py
            if parameter_name == "displacement_field":
                return len(self.u)
            if parameter_name == "force_vector":
                return 3
            raise UnknownParameter()
            # or
            return 1
        ~~~
        """
        return 1



class InferenceProblem:
    def __init__(self):
        self.latent = LatentParameters()
        self.model_errors = OrderedDict()  # key : model_error
        self._noise_models = OrderedDict()  # key : noise_model

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

    def evaluate_model_errors(self, number_vector):
        updated_latent_parameters = self.latent.updated_parameters(number_vector)
        result = {}
        for me_key, prms in updated_latent_parameters.items():
            result[me_key] = self.model_errors[me_key](prms)
        return result

    def set_latent(self, global_and_local_name):
        """
        There is no way to check if a model error actually has a
        `global_and_local_name` parameter, so we pass it to all of them.
        """
        for me_key in self.model_errors:
            self.set_latent_individually(global_and_local_name, me_key)


    def set_latent_individually(self, global_name, model_error_key, local_name=None):
        local_name = global_name if local_name is None else local_name
        model_error = self.model_errors[model_error_key]

        try:
            N = model_error.get_shape(local_name)
        except AttributeError:
            N = 1

        self.latent.add(global_name, local_name, model_error_key, N)


    def loglike(self, number_vector):
        model_errors = self.evaluate_model_errors(number_vector)

        log_like = 0.0
        for noise_key, noise_term in self.noise_models.items():
            log_like += noise_term.loglike_contribution(raw_me)

        return log_like


class VariationalBayesProblem(InferenceProblem, VariationalBayesInterface):
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

    def set_noise_prior(self, name, gamma_or_sd_mean, sd_shape=None):
        if isinstance(gamma_or_sd_mean, Gamma):
            gamma = gamma_or_sd_mean
            assert sd_shape is None
        else:
            sd_shape = sd_shape or 1.0
            gamma = Gamma.FromSD(gamma_or_sd_mean, sd_shape)

        if name not in self.noise_models:
            raise RuntimeError(
                f"{name} is not associated with noise model.. "
                f"Call InferenceProblem.add_noise_model({name}, ...) first."
            )
        self.noise_prior[name] = gamma

    def run(self, **kwargs):
        MVN = self.prior_MVN()
        info = variational_bayes(self, MVN, self.noise_prior, **kwargs)
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
        me = super().evaluate_model_errors(number_vector)

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
            mean, sd = self.prm_prior[name]
            for _ in range(latent.N):
                means.append(mean)
                precs.append(1.0 / sd ** 2)

        return MVN(means, np.diag(precs), name="MVN prior", parameter_names=list(self.latent.keys()))

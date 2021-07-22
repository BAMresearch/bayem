from collections import OrderedDict
from typing import Dict, Hashable

import numpy as np

from .jacobian import jacobian
from .latent import LatentParameters
from .parameters import ParameterList
from .vb import MVN, Gamma, VariationalBayesInterface, variational_bayes


class ModelErrorInterface:
    def __call__(
        self, latent_parameter_list: ParameterList
    ) -> Dict[Hashable, np.ndarray]:
        """
        Evaluate the model error based on the `latent_parameter_list`.
        """
        raise NotImplementedError("Override this!")

    def jacobian(self, latent_parameter_list, w_r_t_what=None):
        return jacobian(self, latent_parameter_list, w_r_t_what)

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
        self.model_errors = OrderedDict()  # key : model_error
        self._noise_models = OrderedDict()  # key : noise_model
        self.latent = LatentParameters(self.model_errors)
        self.latent_noise = LatentParameters(self._noise_models)

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
        """
        Adds a `{key, noise_model}` entry to the inference problem. 
        """

        key = key or f"noise{len(self._noise_models)}"

        assert key not in self._noise_models
        self._noise_models[key] = noise_model
        return key

    def evaluate_model_errors(self, model_error_number_vector):
        updated_latent_parameters = self.latent.updated_parameters(
            model_error_number_vector
        )
        result = {}
        for me_key, prms in updated_latent_parameters.items():
            result[me_key] = self.model_errors[me_key](prms)
        return result

    def loglike(self, number_vector):
        model_error_number_vector = number_vector[: self.latent.vector_length]
        noise_number_vector = number_vector[self.latent.vector_length :]

        model_errors = self.evaluate_model_errors(model_error_number_vector)
        noise_prms = self.latent_noise.updated_parameters(noise_number_vector)

        log_like = 0.0
        for noise_key, noise_term in self.noise_models.items():
            try:
                noise_prm = noise_prms[noise_key]
            except KeyError:
                noise_prm = ParameterList()

            log_like += noise_term.loglike_contribution(model_errors, noise_prm)

        return log_like


class VariationalBayesSolver(VariationalBayesInterface):
    def __init__(self, problem):
        self.problem = problem

    def __call__(self, number_vector, concatenate=True):
        """
        overwrites VariationalBayesInterface.__call__
        """
        me = self.problem.evaluate_model_errors(number_vector)

        errors_by_noise = {}
        for key, noise in self.problem.noise_models.items():
            terms = noise.model_error_terms(me)
            if concatenate:
                terms = np.concatenate(terms)
            errors_by_noise[key] = terms

        return errors_by_noise

    def jacobian(self, number_vector, concatenate=True):
        """
        overwrites VariationalBayesInterface.jacobian
        """
        updated_latent_parameters = self.problem.latent.updated_parameters(
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
                sensor_parameter_jac = jacobian(me, me_parameter_list)

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
        latent_prms = self.problem.latent

        latent_prms.check_priors()
        means = []
        precs = []

        for name, latent in latent_prms.items():
            mean, sd = latent.prior
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
        latent_noise = self.problem.latent_noise
        noise_prior = {}
        for noise_prm_name, latent in latent_noise.items():
            noise_prior[noise_prm_name] = latent.prior
        return noise_prior

    def estimate_parameters(self, **kwargs):
        param0 = self.prior_mvn()
        noise0 = self.prior_noise()
        return variational_bayes(self, param0, noise0, **kwargs)

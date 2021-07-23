from collections import OrderedDict
from typing import Dict, Hashable

import numpy as np

from .jacobian import jacobian_cdf
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
        return jacobian_cdf(self, latent_parameter_list, w_r_t_what)

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

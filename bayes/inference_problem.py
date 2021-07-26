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
        self.latent = LatentParameters()

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
        self.latent.add_model(key, model_error)
        return key

    def add_noise_model(self, noise_model, key=None):
        """
        Adds a `{key, noise_model}` entry to the inference problem. 
        """

        key = key or f"noise{len(self._noise_models)}"

        assert key not in self._noise_models
        self._noise_models[key] = noise_model
        self.latent.add_model(key, noise_model)
        return key
    
    def __call__(self, number_vector):
        updated_latent_parameters = self.latent.updated_parameters(
            number_vector
        )
        return self.evaluate_model_errors(updated_latent_parameters)

    def evaluate_model_errors(self, updated_latent_parameters):
        result = {}
        for me_key, me in self.model_errors.items():
            result[me_key] = me(updated_latent_parameters[me_key])
        return result

    def loglike(self, number_vector):
        updated_latent_parameters = self.latent.updated_parameters(
            number_vector
        )
        model_errors = self.evaluate_model_errors(updated_latent_parameters)

        log_like = 0.0
        for noise_key, noise_term in self.noise_models.items():
            try:
                noise_prm = updated_latent_parameters[noise_key]
            except KeyError:
                noise_prm = ParameterList()

            log_like += noise_term.loglike_contribution(model_errors, noise_prm)

        return log_like


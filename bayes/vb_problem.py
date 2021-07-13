import numpy as np
from .inference_problem import InferenceProblem
from .vb import MVN, Gamma, variational_bayes, VariationalBayesInterface
import logging

logger = logging.getLogger(__name__)


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

                    stacked_jac[:,
                    self.latent[global_name].global_index_range()] += J

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
            mean, sd = self.prm_prior[name]
            for _ in range(latent.N):
                means.append(mean)
                precs.append(1.0 / sd ** 2)

        return MVN(means, np.diag(precs), name="MVN prior",
                   parameter_names=list(self.latent.keys()))

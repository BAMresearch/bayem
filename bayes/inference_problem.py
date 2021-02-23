import numpy as np
from .parameters import ParameterList
from .latent import LatentParameters
from collections import OrderedDict
from .vb import variational_bayes, MVN, Gamma


class ModelError:
    def __call__(self, parameter_list):
        raise NotImplementedError("Override this!")


class NoiseModelInterface:
    def __init__(self):
        self.parameter_list = ParameterList()

    def vector_contribution(self, raw_me):
        raise NotImplementedError()

    def loglike_contribution(self, raw_me, prm):
        raise NotImplementedError()

    def _loglike_term(self, error, sigma):
        return -0.5 * (
            len(error) * np.log(2.0 * np.pi * sigma ** 2)
            + np.sum(np.square(error / sigma ** 2))
        )


class SingleSensorNoise(NoiseModelInterface):
    """
    Noise model with single term for _all_ contributions of the model error.
    """
    def vector_contribution(self, raw_me):
        vector_terms = []
        for exp_me in raw_me.values():
            for sensor_me in exp_me.values():
                vector_terms.append(sensor_me)
        return np.concatenate(vector_terms)



class SingleNoise(NoiseModelInterface):
    """
    Noise model with single term for _all_ contributions of the model error.
    The difference to `SingleSensorNoise` is that each model error is assumed
    to be just a vector instead of a dict with sensor key.
    """
    def vector_contribution(self, raw_me):
        vector_terms = []
        for exp_me in raw_me.values():
            vector_terms.append(exp_me)
        return np.concatenate(vector_terms)



class UncorrelatedNoiseTerm(NoiseModelInterface):
    """
    Uncorrelated noise term that allows to specify exactly which output 
    (defined by key and sensor) from the model error is taken for the term.
    """
    def __init__(self):
        super().__init__()
        self.parameter_list.define("precision")
        self.terms = []

    def add(self, sensor, key=None):
        self.terms.append((sensor, key))

    def vector_contribution(self, raw_me):
        vector_terms = []
        for (sensor, key) in self.terms:
            vector_terms.append(raw_me[key][sensor])
        return np.concatenate(vector_terms)

    def loglike_contribution(self, raw_me, prm):
        error = self.vector_contribution(raw_me)
        return self._loglike_term(error, sigma=1/prm["precision"]**0.5)


class InferenceProblem:
    def __init__(self):
        self.latent = LatentParameters()
        self.model_errors = OrderedDict() # key : model_error
        self.noise_models = OrderedDict() # key : noise_model

    def add_model_error(self, model_error, key=None):
        try:
            model_error.parameter_list
        except AttributeError:
            raise Exception("The model_error _must_ have a .parameter_list!")

        key = key or len(self.model_errors)
        assert key not in self.model_errors

        self.model_errors[key] = model_error
        return key

    def add_noise_model(self, noise_model, key=None):
        try:
            noise_model.parameter_list
        except AttributeError:
            raise Exception("The noise_model _must_ have a .parameter_list!")

        key = key or f"noise{len(self.noise_models)}"

        assert key not in self.noise_models
        self.noise_models[key] = noise_model
        return key

    def __call__(self, number_vector):
        self.latent.update(number_vector)
        result = {}
        for key, me in self.model_errors.items():
            result[key] = me(me.parameter_list)
        return result

    def define_shared_latent_parameter_by_name(self, name):
        for model_error in self.model_errors.values():
            if name in model_error.parameter_list:
                self.latent[name].add(model_error.parameter_list, name)

    def loglike(self, number_vector):
        self.latent.update(number_vector)
        raw_me = {}
        for key, me in self.model_errors.items():
            raw_me[key] = me(me.parameter_list)

        log_like = 0.0
        for noise_key, noise_term in self.noise_models.items():
            log_like += noise_term.loglike_contribution(
                raw_me, noise_term.parameter_list
            )

        return log_like


class VariationalBayesProblem(InferenceProblem):
    def __init__(self):
        super().__init__()
        self.prm_prior = {}
        self.noise_prior = {}

    def set_normal_prior(self, latent_name, mean, sd):
        assert latent_name in self.latent
        self.prm_prior[latent_name] = (mean, sd)

    def set_noise_prior(self, name, sd_mean, sd_scale=1.0):
        assert name in self.noise_models
        self.noise_prior[name] = Gamma.FromSD(sd_mean, sd_scale)

    def _use_default_noise(self):
        if not self.noise_models:
            default = SingleNoise()
            noise_key = self.add_noise_model(default)
            self.noise_prior[noise_key] = Gamma.Noninformative()

    def run(self):
        self._use_default_noise()
        MVN = self.prior_MVN()
        noise = self.prior_noise()
        info = variational_bayes(self, MVN, noise)
        return info

    def __call__(self, number_vector):
        self._use_default_noise()
        me = super().__call__(number_vector)

        errors_by_noise = []
        for noise in self.noise_models.values():
            errors_by_noise.append(noise.vector_contribution(me))

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
        return list(self.noise_prior.values())

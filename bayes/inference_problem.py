import numpy as np
from .parameters import LatentParameters, ModelErrorParameters
from collections import OrderedDict
from .vb import variational_bayes, MVN, Gamma, noninformative_gamma_prior


class ModelError:
    def __call__(self, parameter_list):
        raise NotImplementedError("Override this!")


class SingleSensorNoise:
    def define_parameters(self):
        p = ModelErrorParameters()
        p.define("sigma")
        return p

    def vector_contribution(self, raw_me):
        vector_terms = []
        for exp_me in raw_me.values():
            for sensor_me in exp_me.values():
                vector_terms.append(sensor_me)
        return np.concatenate(vector_terms)

class SingleNoise:
    def define_parameters(self):
        p = ModelErrorParameters()
        p.define("sigma")
        return p

    def vector_contribution(self, raw_me):
        vector_terms = []
        for exp_me in raw_me.values():
            vector_terms.append(exp_me)
        return np.concatenate(vector_terms)

class NoiseTerm:
    def __init__(self, sensor=None, key=None):
        if sensor is None:
            self.terms = []
        else:
            self.terms = [(sensor, key)]

    def add(self, sensor, key=None):
        self.terms.append((sensor, key))

    def define_parameters(self):
        p = ModelErrorParameters()
        p.define("precision")
        return p

    def vector_contribution(self, raw_me):
        vector_terms = []
        for (sensor, key) in self.terms:
            vector_terms.append(raw_me[key][sensor])
        return np.concatenate(vector_terms)

    def loglike_contribution(self, raw_me, prm):
        error = self.vector_contribution(raw_me)
        return -0.5 * (
            len(error) * np.log(2.0 * np.pi / prm["precision"])
            + np.sum(np.square(error * prm["precision"]))
        )


class InferenceProblem:
    def __init__(self):
        self.latent = LatentParameters()
        self.model_errors = OrderedDict()
        self.noise_models = OrderedDict()

    def add_model_error(self, model_error, parameter_list, key=None):
        key = key or len(self.model_errors)
        assert key not in self.model_errors

        self.model_errors[key] = model_error
        self.latent.define_parameter_list(parameter_list, key)

        return key

    def add_noise_model(self, noise_model, parameter_list=None, key=None):
        parameter_list = parameter_list or noise_model.define_parameters()
        key = key or f"noise{len(self.noise_models)}"

        assert key not in self.noise_models
        self.noise_models[key] = noise_model
        self.latent.define_parameter_list(parameter_list, key)
        return key

    def __call__(self, number_vector):
        prm_lists = self.latent.update(number_vector)
        result = {}
        for key, me in self.model_errors.items():
            result[key] = me(prm_lists[key])
        return result

    def loglike(self, number_vector):
        prm_lists = self.latent.update(number_vector)
        raw_me = {}
        for key, me in self.model_errors.items():
            raw_me[key] = me(prm_lists[key])

        log_like = 0.
        for noise_key, noise_term in self.noise_models.items():
            log_like += noise_term.loglike_contribution(raw_me, prm_lists[noise_key])

        return log_like




def vb_wrap(me, param0, noise0):
    def f(number_vector):
        errors_by_noise = me(number_vector)
        return np.concatenate(list(errors_by_noise.values()))

    errors_by_noise = me(param0.mean)
    f.noise_pattern = []
    i = 0
    for error in errors_by_noise.values():
        N = len(error)
        f.noise_pattern.append(list(range(i, i + N)))
        i += N

    return variational_bayes(f, param0, noise0)


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
            noise_key = self.add_noise_model(default, default.define_parameters())
            self.noise_prior[noise_key] = noninformative_gamma_prior(1)

    def run(self):
        self._use_default_noise()
        MVN = self.prior_MVN()
        noise = self.prior_noise()
        info = vb_wrap(self, MVN, noise)
        return info

    def __call__(self, number_vector):
        self._use_default_noise()
        me = super().__call__(number_vector)

        errors_by_noise = OrderedDict()
        for noise_name, noise in self.noise_models.items():
            errors_by_noise[noise_name] = noise.vector_contribution(me)

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
        scales = []
        shapes = []
        for gamma in self.noise_prior.values():
            assert len(gamma.c) == 1
            scales.append(gamma.c[0])
            shapes.append(gamma.s[0])

        return Gamma(c=scales, s=shapes)

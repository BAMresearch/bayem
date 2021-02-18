import numpy as np
from .parameters import *
from .vb import *

class ModelError:
    def __call__(self, parameter_list):
        raise NotImplementedError("Override this!")

class InferenceProblem:
    def __init__(self):
        self.latent = LatentParameters()
        self.model_errors = OrderedDict()

    def add_model_error(self, model_error, parameter_list, key=None):
        key = key or len(self.model_errors)
        assert key not in self.model_errors

        self.model_errors[key] = model_error
        self.latent.define_parameter_list(parameter_list, key)

        return key

    def __call__(self, number_vector):
        prm_lists = self.latent.update(number_vector)
        result = {}
        for key, me in self.model_errors.items():
            result[key] = me(prm_lists[key])
        return result

class NoiseGroup(set):
    def __init__(self, gamma):
        self.gamma = gamma
        
def vb_wrap(me, param0, noise0):

    def f(number_vector):
        errors_by_noise = me(number_vector)
        return np.concatenate(list(errors_by_noise.values()))

    if me.noise_groups:
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
        self.noise_groups = OrderedDict()

    def set_normal_prior(self, latent_name, mean, sd):
        assert latent_name in self.latent
        self.latent[latent_name].vb_prior = (mean, sd)
    
    def define_noise_group(self, name, sd_mean, sd_scale=1.):
        assert name not in self.noise_groups
        self.noise_groups[name] = NoiseGroup(Gamma.FromSD(sd_mean, sd_scale))

    def add_to_noise_group(self, name, sensor, key=None):
        assert name in self.noise_groups
        self.noise_groups[name].add((key, sensor))

    def run(self, **kwargs):
        MVN = self.prior_MVN()
        noise = self.prior_noise()
        info = vb_wrap(self, MVN, noise)
        return info

    def __call__(self, number_vector):
        prm_lists = self.latent.update(number_vector)
        me = {}
        for key, f in self.model_errors.items():
            me[key] = f(prm_lists[key])

        errors_by_noise = OrderedDict()
        if self.noise_groups:
            for noise_name, noise in self.noise_groups.items():
                errors = []
                for (key, sensor) in noise:
                    errors.append(me[key][sensor])
                errors_by_noise[noise_name] = np.concatenate(errors)
        else:
            errors = []
            for m in me.values():
                for s in m.values():
                    errors.append(s)
            errors_by_noise[None] = np.concatenate(errors)

        return errors_by_noise
        

    def prior_MVN(self):
        from bayes.vb import MVN

        means = []
        precs = []

        for latent in self.latent.values():
            for _ in range(latent.N):
                mean, sd = latent.vb_prior
                means.append(mean)
                precs.append(1.0 / sd ** 2)

        return MVN(means, np.diag(precs))

    def prior_noise(self):
        from bayes.vb import Gamma

        if not self.noise_groups:
            return None

        scales = []
        shapes = []
        for noise in self.noise_groups.values():
            gamma = noise.gamma
            assert len(gamma.c) == 1
            scales.append(gamma.c[0])
            shapes.append(gamma.s[0])

        return Gamma(c=scales, s=shapes)

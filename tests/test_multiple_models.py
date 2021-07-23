import numpy as np
import unittest
import scipy.stats
from bayes.vb import *
from bayes.parameters import ParameterList
from bayes.inference_problem import (
    ModelErrorInterface,
    InferenceProblem,
)
from bayes.solver import VariationalBayesSolver
from bayes.noise import UncorrelatedSingleNoise

np.random.seed(6174)

A1, B1, A2, B2 = 100.0, 200.0, 300.0, 400.0
noise_sd = 12.0

N = 2000
xs = np.linspace(0, 1, N)

data_1 = A1 * xs + B1 + np.random.normal(0, noise_sd, N)
data_2 = A2 * xs + B2 + np.random.normal(0, noise_sd, N)

"""
Combining two linear models can be done _naively_ by manually defining the 
four parameters prm[0..3] and hardcode two separate model errors. In 
`multi_me`, they are both evaluated and concatenated.
"""


def model_error_1(prm):
    return prm[0] * xs + prm[1] - data_1


def model_error_2(prm):
    return prm[2] * xs + prm[3] - data_2


def multi_me(prm):
    return np.append(model_error_1(prm), model_error_2(prm))


"""
Both logically (why have two models instead of one with different parameters) 
and practially (imagine having > 100 models), this is bad. Instead, we want to
define the model once. Bonus: We want to name the parameters. e.g. "B" instead 
of index 1. Goto "test_joint" to see that in action.
"""


def model(prm):
    return prm["A"] * xs + prm["B"]


class ModelError(ModelErrorInterface):
    def __init__(self, fw, data):
        self.fw, self.data = fw, data

    def __call__(self, latent_prm):
        return {"dummy_sensor": self.fw(latent_prm) - self.data}


class Test_VB(unittest.TestCase):
    def check_posterior(self, info, noise_key=None):
        param_post, noise_post = info.param, info.noise
        for i, param_true in enumerate([A1, B1, A2, B2]):
            posterior_mean = param_post.mean[i]
            posterior_std = param_post.std_diag[i]

            self.assertLess(posterior_std, 4)
            self.assertAlmostEqual(posterior_mean, param_true, delta=2 * posterior_std)

        if noise_key is None:
            post_noise_precision = noise_post.mean
        else:
            post_noise_precision = noise_post[noise_key].mean

        post_noise_std = 1.0 / post_noise_precision ** 0.5
        self.assertAlmostEqual(post_noise_std, noise_sd, delta=noise_sd / 100)

        self.assertLess(info.nit, 20)

    def test_multiple(self):

        prior_mean = np.r_[A1, B1, A2, B2] + 0.5  # slightly off
        prior_prec = np.r_[0.25, 0.25, 0.25, 0.25]
        prior = MVN(prior_mean, np.diag(prior_prec))

        info = variational_bayes(multi_me, prior)
        self.check_posterior(info)

    def test_joint_evaluate(self):
        # Define two ModelErrors, but note that both use the same model.
        me1 = ModelError(model, data_1)
        me2 = ModelError(model, data_2)

        # For the inference, we combine them and use a 'key' to distinguish
        # e.g. "A" from the one model to "A" from the other one.
        problem = InferenceProblem()
        key1 = problem.add_model_error(me1)
        key2 = problem.add_model_error(me2)

        problem.latent["B1"].add(me1, "B")
        problem.latent["B2"].add(me2, "B")
        problem.latent["A"].add_shared()

        parameter_vec = np.array([1, 2, 4])
        error_list = problem.evaluate_model_errors(parameter_vec)
        error_multi = multi_me([4, 1, 4, 2])

        error_concatentated = np.concatenate(
            [error_list[key1]["dummy_sensor"], error_list[key2]["dummy_sensor"]]
        )
        np.testing.assert_almost_equal(error_concatentated, error_multi)

    def test_joint(self):
        # Define two ModelErrors, but note that both use the same model.
        me1 = ModelError(model, data_1)
        me2 = ModelError(model, data_2)

        # For the inference, we combine them and use a 'key' to distinguish
        # e.g. "A" from the one model to "A" from the other one.
        problem = InferenceProblem()
        problem.add_model_error(me1)
        problem.add_model_error(me2)

        problem.latent["A1"].add(me1, "A")
        problem.latent["B1"].add(me1, "B")
        problem.latent["A2"].add(me2, "A")
        problem.latent["B2"].add(me2, "B")

        problem.latent["A1"].prior = scipy.stats.norm(loc=A1 + 0.5, scale=2)
        problem.latent["B1"].prior = scipy.stats.norm(loc=B1 + 0.5, scale=2)
        problem.latent["A2"].prior = scipy.stats.norm(loc=A2 + 0.5, scale=2)
        problem.latent["B2"].prior = scipy.stats.norm(loc=B2 + 0.5, scale=2)

        noise_key = problem.add_noise_model(UncorrelatedSingleNoise())
        problem.latent_noise[noise_key].add(noise_key, "precision")
        problem.latent_noise[noise_key].prior = scipy.stats.gamma(1, 1)

        vbs = VariationalBayesSolver(problem)
        info = vbs.estimate_parameters()
        print(info)
        self.check_posterior(info, noise_key)


if __name__ == "__main__":
    unittest.main()

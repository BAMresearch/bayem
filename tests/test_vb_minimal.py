import unittest
import numpy as np
from bayes.vb import *

np.random.seed(6174)
x = np.linspace(0, 1, 1000)
A, B, sd = 7.0, 42.0, 0.1
data = A * x + B + np.random.normal(0, sd, len(x))


def me_dict(parameters):
    return {"noise": parameters[0] * x + parameters[1] - data}


def me_vector(parameters):
    return parameters[0] * x + parameters[1] - data


class Test_VB(unittest.TestCase):
    def run_vb(self, model_error):

        param_prior = MVN([6, 11], [[1 / 3 ** 2, 0], [0, 1 / 3 ** 2]])
        noise_prior = Gamma(shape=0.1, scale=1000)

        info = variational_bayes(model_error, param_prior, noise_prior)
        param_post, noise_post = info.param, info.noise
        print(info)

        # if plot:
        #     plot_pdf(
        #         param_post,
        #         expected_value=param_true,
        #         compare_with=param_prior,
        #         plot="joint",
        #     )
        #
        for i, correct_value in enumerate([A, B]):
            posterior_mean = param_post.mean[i]
            posterior_std = param_post.std_diag[i]

            self.assertLess(posterior_std, 0.3)
            self.assertAlmostEqual(
                posterior_mean, correct_value, delta=2 * posterior_std
            )

        post_noise_precision = noise_post.mean
        post_noise_std = 1.0 / post_noise_precision ** 0.5
        self.assertAlmostEqual(post_noise_std, sd, delta=sd / 100)

        self.assertLess(info.nit, 20)

    def test_dict(self):
        self.run_vb(me_dict)

    def test_vector(self):
        self.run_vb(me_vector)


if __name__ == "__main__":
    unittest.main()

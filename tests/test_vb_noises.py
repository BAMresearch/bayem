import numpy as np
import unittest
from bayes.vb import *

PRM_A, PRM_B = 7.0, 10.0  # slope and offset to identify
NOISE0_SD = 0.1
NOISE1_SD = 2.0
N = 500


class ForwardModel:
    def __init__(self):
        self.x = np.linspace(0, 1, N)

    def __call__(self, all_prms):
        m, c = all_prms
        return {
            "group0": c + m * self.x[0 : int(N / 2)],
            "group1": c + m * self.x[int(N / 2) : N],
        }


class ModelError:
    def __init__(self, fw, data):
        self._fw = fw
        self._data = data

    def __call__(self, prms):
        me = self._fw(prms)
        me["group0"] -= self._data["group0"]
        me["group1"] -= self._data["group1"]
        return me


def to_mvn(parameters):
    n = len(parameters)
    means = np.zeros(n)
    precisions = np.zeros((n, n))
    for i, (mean, sd) in enumerate(parameters):
        means[i] = mean
        precisions[i, i] = 1.0 / sd ** 2
    return MVN(means, precisions)


class TestTwoNoises(unittest.TestCase):
    def run_test(self, noise_prior, delta):
        """
        Infers two parameters of a linear model where the data is arranged
        in two noise groups.

        noise_prior:
            gamma distribution for the noise precision or
            None for a noninformative prior
        delta:
            absolute value of the inferred noise standard deviation to compare
        """
        np.random.seed(42)

        fw = ForwardModel()
        data = fw([PRM_A, PRM_B])

        data["group0"] += np.random.normal(0, NOISE0_SD, len(data["group0"]))
        data["group1"] += np.random.normal(0, NOISE1_SD, len(data["group1"]))

        param_prior = to_mvn([(6.0, 2.0), (15.0, 7.0)])

        me = ModelError(fw, data)
        info = variational_bayes(me, param_prior, noise_prior)

        param = info.param
        self.assertAlmostEqual(param.mean[0], PRM_A, delta=2 * param.std_diag[0])
        self.assertAlmostEqual(param.mean[1], PRM_B, delta=2 * param.std_diag[1])

        noise_sds = {n: 1.0 / gamma.mean ** 0.5 for n, gamma in info.noise.items()}
        self.assertAlmostEqual(noise_sds["group0"], NOISE0_SD, delta=delta)
        self.assertAlmostEqual(noise_sds["group1"], NOISE1_SD, delta=delta)
        print("noise prior was", noise_prior)
        print(param)
        print(noise_sds)
        print()

    def test_proper_noise(self):
        """Use a noise_prior that is based on the actual values"""
        noise_prior = {}
        noise_prior["group0"] = Gamma.FromSDQuantiles(0.1 * NOISE0_SD, 10 * NOISE0_SD)
        noise_prior["group1"] = Gamma.FromSDQuantiles(0.1 * NOISE1_SD, 10 * NOISE1_SD)
        self.run_test(noise_prior, delta=0.05)

    def test_noninformative_noise(self):
        """The noninformative prior requires a higher delta to pass the test"""
        self.run_test(None, delta=0.1)

    def test_wrong_number_of_noises(self):
        def me(prm):
            return np.array(prm - 10)

        # That should work:
        correct_noise = Gamma(1, 2)
        info = variational_bayes(me, MVN(7, 12), correct_noise)
        self.assertTrue(info.success)

        # Providing a wrong dimension in the noise pattern should fail.
        wrong_noise = Gamma((1, 1), (2, 2))
        self.assertRaises(Exception, variational_bayes, me, MVN(7, 12), wrong_noise)


if __name__ == "__main__":
    unittest.main()

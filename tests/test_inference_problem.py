import numpy as np
import unittest
import scipy.optimize

from bayes.vb import Gamma
from bayes.parameters import ParameterList
from bayes.noise import UncorrelatedSingleNoise
from bayes.inference_problem import InferenceProblem


def dummy_model_error(prms):
    x = np.r_[1, 2]
    return {"dummy_sensor": x * prms["B"] - 20}


class TestProblem(unittest.TestCase):
    def test_add(self):
        p = InferenceProblem()
        p.add_model_error(dummy_model_error, key="0")
        with self.assertRaises(Exception):
            p.add_model_error(dummy_model_error, key="0")

    def test_latent_parameters(self):
        p = InferenceProblem()
        me_key = p.add_model_error(dummy_model_error)
        p.latent["B"].add_shared()

        me = p([42])
        self.assertListEqual([22, 64], list(me[me_key]["dummy_sensor"]))

    def test_shared_latent_evaluate(self):
        p = InferenceProblem()
        N = 3
        for _ in range(N):
            p.add_model_error(dummy_model_error)
        p.latent["B"].add_shared()
        self.assertEqual(len(p.latent["B"]), N)

        result = p([42])
        for key, me in p.model_errors.items():
            self.assertListEqual(list(result[key]), list(dummy_model_error({"B": 42})))

    def test_maximum_likelihood(self):
        p = InferenceProblem()
        p.add_model_error(dummy_model_error)
        p.latent["B"].add_shared()
        noise = UncorrelatedSingleNoise()
        noise.parameter_list["precision"] = 1.0
        p.add_noise_model(noise)

        def minus_loglike(x):
            return -p.loglike([x])

        result = scipy.optimize.minimize_scalar(minus_loglike)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.x, 12)  # 12. Trust me! :P


if __name__ == "__main__":
    unittest.main()

import numpy as np
import unittest
import scipy.optimize

from bayes.vb import Gamma
from bayes.parameters import ParameterList
from bayes.noise import UncorrelatedSingleNoise
from bayes.inference_problem import VariationalBayesProblem, InferenceProblem


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

        me = p.evaluate_model_errors([42])
        self.assertListEqual([22, 64], list(me[me_key]["dummy_sensor"]))

    def test_shared_latent_evaluate(self):
        p = InferenceProblem()
        N = 3
        for _ in range(N):
            p.add_model_error(dummy_model_error)
        p.latent["B"].add_shared()
        self.assertEqual(len(p.latent["B"]), N)

        result = p.evaluate_model_errors([42])
        for key, me in p.model_errors.items():
            self.assertListEqual(list(result[key]), list(dummy_model_error({"B": 42})))

    def test_maximum_likelihood(self):
        p = InferenceProblem()
        p.add_model_error(dummy_model_error)
        p.latent["B"].add_shared()
        noise = UncorrelatedSingleNoise()
        noise.parameter_list["precision"] = 1.
        p.add_noise_model(noise)
        
        def minus_loglike(x):
            return -p.loglike([x])

        result = scipy.optimize.minimize_scalar(minus_loglike)
        self.assertTrue(result.success) 
        self.assertAlmostEqual(result.x, 12)  # 12. Trust me! :P



class TestVBProblem(unittest.TestCase):
    def test_prior(self):
        p = VariationalBayesProblem()
        p.add_model_error(dummy_model_error)
        p.latent["B"].add_shared()
        p.set_normal_prior("B", 0.0, 1.0)
        self.assertRaises(Exception, p.set_normal_prior, "not B", 0.0, 1.0)

        self.assertRaises(Exception, p.set_noise_prior, "noise", 1.0, 1.0)
        p.add_noise_model(UncorrelatedSingleNoise(), key="noise")
        p.latent_noise["noise"].add("noise")

        p.set_noise_prior("noise", Gamma.Noninformative())

        result = p([0.1])
        self.assertEqual(len(result), 1)  # one noise group
        self.assertEqual(len(result["noise"]), 2)

        jac = p.jacobian([0.1])


if __name__ == "__main__":
    unittest.main()

import numpy as np
import unittest
from bayes.vb import Gamma
from bayes.parameters import ParameterList
from bayes.noise import UncorrelatedSingleNoise
from bayes.inference_problem import VariationalBayesProblem, InferenceProblem


def dummy_model_error(prms):
    x = np.r_[1, 2]
    return {"dummy_sensor": x * prms["B"]}


class TestProblem(unittest.TestCase):
    def test_add(self):
        p = InferenceProblem()
        p.add_model_error(dummy_model_error, key="0")
        with self.assertRaises(Exception):
            p.add_model_error(dummy_model_error, key="0")

    def test_latent_parameters(self):
        p = InferenceProblem()
        me_key = p.add_model_error(dummy_model_error)
        p.set_latent_individually("B", me_key, "B")
        # or
        p.set_latent_individually("B", me_key)

        me = p.evaluate_model_errors([42])
        self.assertListEqual([42, 84], list(me[me_key]["dummy_sensor"]))

    def test_shared_latent_evaluate(self):
        p = InferenceProblem()
        N = 3
        for _ in range(N):
            p.add_model_error(dummy_model_error)
        p.set_latent("B")
        self.assertEqual(len(p.latent["B"]), N)

        result = p.evaluate_model_errors([42])
        for key, me in p.model_errors.items():
            self.assertListEqual(list(result[key]), list(dummy_model_error({"B": 42})))


class TestVBProblem(unittest.TestCase):
    def test_prior(self):
        p = VariationalBayesProblem()
        p.add_model_error(dummy_model_error)
        p.set_latent("B")
        p.set_normal_prior("B", 0.0, 1.0)
        self.assertRaises(Exception, p.set_normal_prior, "not B", 0.0, 1.0)

        self.assertRaises(Exception, p.set_noise_prior, "noise", 1.0, 1.0)
        p.add_noise_model(UncorrelatedSingleNoise(), key="noise")
        p.set_noise_prior("noise", 1.0, 1.0)
        p.set_noise_prior("noise", Gamma.Noninformative())

        result = p([0.1])
        self.assertEqual(len(result), 1)  # one noise group
        self.assertEqual(len(result["noise"]), 2)


if __name__ == "__main__":
    unittest.main()

import numpy as np
import unittest
import scipy.stats
from bayes.vb import Gamma
from bayes.parameters import ParameterList
from bayes.noise import UncorrelatedSingleNoise
from bayes.inference_problem import InferenceProblem, VariationalBayesProblem


class ModelError:
    def __init__(self):
        self.parameter_list = ParameterList()
        self.parameter_list.define("B")

    def __call__(self):
        x = np.linspace(0, 1, 10)
        return {"dummy_sensor": x * self.parameter_list["B"]}


class TestProblem(unittest.TestCase):
    def test_add(self):
        p = InferenceProblem()
        me = ModelError()
        p.add_model_error(me, key="0")
        self.assertRaises(Exception, p.add_model_error, me, key="0")

    def test_shared_latent_evaluate(self):
        p = InferenceProblem()
        N = 3
        for _ in range(N):
            p.add_model_error(ModelError())
        p.define_shared_latent_parameter_by_name("B")
        self.assertEqual(len(p.latent["B"]), N)

        result = p([0.1])
        for key, model_error in p.model_errors.items():
            self.assertAlmostEqual(model_error.parameter_list["B"], 0.1)
            self.assertListEqual(list(result[key]), list(model_error()))


class TestVBProblem(unittest.TestCase):
    def test_prior(self):
        p = InferenceProblem()
        key = p.add_model_error(ModelError())
        p.define_shared_latent_parameter_by_name("B")
        p.set_parameter_prior_normal("B", 0.0, 1.0)
        self.assertAlmostEqual(p.prm_prior["B"].mean(), 0.0)
        self.assertAlmostEqual(p.prm_prior["B"].std(), 1.0)
        self.assertRaises(Exception, p.set_parameter_prior, "not B", 0.0, 1.0)

        self.assertRaises(Exception, p.set_noise_precision_prior_sd, "noise", 1.0, 1.0)
        p.add_noise_model(UncorrelatedSingleNoise(), key="noise")
        p.set_noise_precision_prior_sd("noise", 1.0, 1.0)

        result = p([0.1])
        self.assertEqual(len(result[key]["dummy_sensor"]), 10)

    def test_wrong_prior_type(self):
        p = VariationalBayesProblem()
        p.add_model_error(ModelError())
        p.define_shared_latent_parameter_by_name("B")
        p.set_parameter_prior("B", scipy.stats.crystalball(42, 6174))
        with self.assertRaises(Exception) as e:
            p.prior_MVN()
        print("Expected exception\n\t", e.exception)
        
        with self.assertRaises(Exception) as e:
            p.set_parameter_prior("B", "what?")
        print("Expected exception\n\t", e.exception)

        p.add_noise_model(UncorrelatedSingleNoise(), key="noise")
        p.set_noise_precision_prior_sd("noise", 6.174, 42.0)
        mean, var = p.noise_prior["noise"].mean(), p.noise_prior["noise"].var()
        scale = var / mean
        shape = mean / scale

        self.assertAlmostEqual(mean, 1.0 / 6.174 ** 2)
        self.assertAlmostEqual(shape, 42.0)


if __name__ == "__main__":
    unittest.main()

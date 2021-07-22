import numpy as np
import unittest
import bayes.noise
import bayes.parameters
from copy import deepcopy
import scipy.stats

CHECK = np.testing.assert_almost_equal  # just to make it shorter

"""
Example:
========
    See test/test_noise.py for an exact implementation of the following example.

    Assume that we have measured forces on a structure and the room temperature
    and want to use it to infer the parameters of a numerical model.

    The two independent data sets are added to two model_errors (with 
    model_error_keys 'exp1' and 'exp2'). 
"""

exp1_me = {"ForceSensor": np.r_[0.9, 2.0, 3.1], "TemperatureSensor": np.r_[22.1]}
exp2_me = {"ForceSensor": np.r_[1.1, 1.9, 3.0], "TemperatureSensor": np.r_[22.7]}

"""
    Thus, the example inference problem
    would return the 
"""

model_error_dict = {"exp1": exp1_me, "exp2": exp2_me}
print(f"{model_error_dict = }")

"""
    The ForceSensor measurements of both measurements should now be in the same
    noise group and we expect it to be
"""
expected_force_terms = [exp1_me["ForceSensor"], exp2_me["ForceSensor"]]
print(f"{expected_force_terms = }")

"""
    The TemperatureSensor (maybe we changed the placement between
    experiments) are assumed to have individual noise models.

    Lets test:
"""


class TestNoiseTerms(unittest.TestCase):
    def test_uncorrelated_noise(self):
        n_force = bayes.noise.UncorrelatedNoiseModel()
        n_force.add("exp1", "ForceSensor")
        n_force.add("exp2", "ForceSensor")

        n_temp1 = bayes.noise.UncorrelatedNoiseModel()
        n_temp1.add("exp1", "TemperatureSensor")

        n_temp2 = bayes.noise.UncorrelatedNoiseModel()
        n_temp2.add("exp2", "TemperatureSensor")

        force_terms = n_force.model_error_terms(model_error_dict)
        CHECK(force_terms, expected_force_terms)

        temp1_terms = n_temp1.model_error_terms(model_error_dict)
        CHECK(temp1_terms[0], exp1_me["TemperatureSensor"])

        temp2_terms = n_temp2.model_error_terms(model_error_dict)
        CHECK(temp2_terms[0], exp2_me["TemperatureSensor"])

        """
        We can also rearrange the individual terms back to the
        nested `model_error_key`-`sensor`-dict:
        """

        by_keys = n_force.by_keys(force_terms)
        self.assertListEqual(list(by_keys), ["exp1", "exp2"])
        self.assertEqual(len(by_keys["exp1"]), 1)
        self.assertEqual(len(by_keys["exp2"]), 1)

        CHECK(by_keys["exp1"]["ForceSensor"], model_error_dict["exp1"]["ForceSensor"])
        CHECK(by_keys["exp2"]["ForceSensor"], model_error_dict["exp2"]["ForceSensor"])

        """
        Note that the "by_keys" should fail, if we provide the wrong number
        of terms
        """
        with self.assertRaises(Exception) as e:
            n_force.by_keys([[1, 2, 3]])
        print(f"Expected exception:\n {e.exception = }")


"""
A more convenient way to define noise models for the same sensors is to use
the UncorrelatedSensorNoise that slightly shortens the force noise definition
above.
"""

class TestNoiseSensor(unittest.TestCase):
    def test_uncorrelated_noise(self):
        n_force = bayes.noise.UncorrelatedSensorNoise("ForceSensor")
        force_terms = n_force.model_error_terms(model_error_dict)
        CHECK(force_terms, expected_force_terms)

        """
        Note that you can also provide a list of sensors, if you want to
        add multiple sensors in one noise term.
        """
        bayes.noise.UncorrelatedSensorNoise(["ForceSensor", "TemperatureSensor"])

"""
The simplest case of having only a single noise model is covered by the
UncorrelatedSingleNoise that combines all terms.
"""

class TestNoiseSingle(unittest.TestCase):
    def test_uncorrelated_noise(self):
        n_single = bayes.noise.UncorrelatedSingleNoise()
        terms = n_single.model_error_terms(model_error_dict)
        self.assertEqual(len(terms), 4)

"""
Note that the behavoir of "Jacobian" is identical to the model errors
"""

class TestNoiseJacobian(unittest.TestCase):
    def test_jacobian(self):
        J1 = np.array([[1, 2], [3, 4], [5, 6]])
        J2 = np.array([[10, 20], [30, 40], [50, 60]])
        jacobian_dict = {
            "exp1": {"ForceSensor": J1},
            "exp2": {"ForceSensor": J2},
        }
        n = bayes.noise.UncorrelatedSingleNoise()
        j = n.jacobian_terms(jacobian_dict)

        CHECK(j, [J1, J2])

"""
The loglikelihood function is not properly tested, but can be evaluated
at your own risk
"""

class TestNoiseLoglike(unittest.TestCase):
    def test_loglike(self):
        n = bayes.noise.UncorrelatedSingleNoise()
        p = bayes.parameters.ParameterList()
        p.define("precision")

        for precision in np.geomspace(1e-4, 1e4, 10):
            p["precision"] = precision

            ll_ours = n.loglike_contribution(model_error_dict, p)

            ll_scipy = 0.
            terms = n.model_error_terms(model_error_dict)
            for term in terms: 
                ll_scipy += sum(scipy.stats.norm.logpdf(term, scale=1./precision**0.5))
       
            self.assertAlmostEqual(ll_ours, ll_scipy)

if __name__ == "__main__":
    unittest.main()

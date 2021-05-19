import numpy as np
import unittest
from bayes.noise import *
from copy import deepcopy

CHECK = np.testing.assert_almost_equal  # just to make it shorter

truss_model_error = {"ForceSensor": np.r_[1, 2, 3], "TemperatureSensor": np.r_[20]}
beam_model_error = {"ForceSensor": np.r_[10, 20, 30], "InclinationSensor": np.r_[0.42]}

model_error_dict = {"truss_key": truss_model_error, "beam_key": beam_model_error}


class TestNoise(unittest.TestCase):
    def test_uncorrelated_noise(self):
        n = UncorrelatedNoiseModel()
        n.add("ForceSensor", "truss_key")
        v = n.vector_contribution(model_error_dict)
        CHECK(v, [1, 2, 3])

        n.add("ForceSensor", "beam_key")
        v = n.vector_contribution(model_error_dict)
        CHECK(v, [1, 2, 3, 10, 20, 30])

    def test_single_noise(self):
        n = UncorrelatedSingleNoise()
        v = n.vector_contribution(model_error_dict)
        # the actual ordering should not matter, as long as it is consistent.
        somehow_expected = [1, 2, 3, 20, 10, 20, 30, 0.42]
        CHECK(np.sort(v), np.sort(somehow_expected))

    def test_split(self):
        n = UncorrelatedSingleNoise()

        with self.assertRaises(Exception) as e:
            n.split([1, 2, 3])  # not evaluated
        print("Expected exception:\n", e.exception)

        v = n.vector_contribution(model_error_dict)

        with self.assertRaises(Exception) as e:
            n.split([1, 2, 3])  # wrong length of vector
        print("Expected exception:\n", e.exception)

        splitted = n.split(v)
        self.check_nested_dict(splitted, model_error_dict)

    def check_nested_dict(self, first, second):
        self.assertListEqual(list(first.keys()), list(second.keys()))
        for me_key in first:
            self.assertListEqual(
                list(first[me_key].keys()), list(second[me_key].keys())
            )
            for sensor_key in first[me_key]:
                CHECK(first[me_key][sensor_key], second[me_key][sensor_key])

    def test_sensor_noise(self):
        n = UncorrelatedSensorNoise("ForceSensor")
        v = n.vector_contribution(model_error_dict)
        # the actual ordering should not matter, as long as it is consistent.
        somehow_expected = [1, 2, 3, 10, 20, 30]
        CHECK(np.sort(v), np.sort(somehow_expected))

    def test_sensor_noise_multiple(self):
        n = UncorrelatedSensorNoise(["ForceSensor", "InclinationSensor"])
        v = n.vector_contribution(model_error_dict)
        # the actual ordering should not matter, as long as it is consistent.
        somehow_expected = [1, 2, 3, 10, 20, 30, 0.42]
        CHECK(np.sort(v), np.sort(somehow_expected))

        splitted = n.split(v)

        expected = deepcopy(model_error_dict)
        # remove "TemperatureSensor"
        expected["truss_key"].pop("TemperatureSensor")
        self.check_nested_dict(splitted, expected)

    def test_jacobian(self):
        J1 = np.array([[1, 2], [3, 4], [5, 6]])
        J2 = np.array([[10, 20], [30, 40], [50, 60]])
        jacobian_dict = {
            "truss_key": {"ForceSensor": J1},
            "beam_key": {"ForceSensor": J2}
        }
        n = UncorrelatedSingleNoise()
        j = n.jacobian_contribution(jacobian_dict)

        CHECK(j, np.vstack([J1, J2]))
        splitted = n.split(j)
        self.check_nested_dict(splitted, jacobian_dict)


if __name__ == "__main__":
    unittest.main()

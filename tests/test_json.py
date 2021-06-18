import unittest
import numpy as np
import bayes.vb
import json


class TestJSON(unittest.TestCase):
    def test_json(self):
        data = {}
        data["parameter_prior"] = bayes.vb.MVN(
            mean=np.r_[1, 2, 3],
            precision=np.diag([1, 2, 3]),
            parameter_names=["A", "B", "C"],
        )

        data["noise_prior"] = bayes.vb.Gamma()
        data["non bayes thing"] = {"best number": 6174.0}

        string = json.dumps(data, cls=bayes.vb.BayesEncoder, indent=2)
        print(string)

        loaded = json.loads(string, object_hook=bayes.vb.bayes_hook)

        A, B = data["parameter_prior"], loaded["parameter_prior"]
        CHECK = np.testing.assert_array_equal
        self.assertEqual(A.name, B.name)
        self.assertEqual(A.parameter_names, B.parameter_names)
        CHECK(A.mean, B.mean)
        CHECK(A.precision, B.precision)

        C, D = data["noise_prior"], loaded["noise_prior"]
        self.assertEqual(C.name, D.name)
        self.assertEqual(C.shape, D.shape)
        self.assertEqual(C.scale, D.scale)


if __name__ == "__main__":
    unittest.main()

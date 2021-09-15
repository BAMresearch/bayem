import unittest
import numpy as np
import bayem
import json


class TestJSON(unittest.TestCase):
    def test_json(self):
        data = {}
        data["parameter_prior"] = bayem.MVN(
            mean=np.r_[1, 2, 3],
            precision=np.diag([1, 2, 3]),
            parameter_names=["A", "B", "C"],
        )

        data["noise_prior"] = bayem.Gamma()
        data["non bayem thing"] = {"best number": 6174.0}

        string = json.dumps(data, cls=bayem.BayemEncoder, indent=2)
        print(string)

        loaded = json.loads(string, object_hook=bayem.bayem_hook)

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

    def test_vb_result(self):
        def dummy_me(prm):
            return prm ** 2

        result = bayem.variational_bayes(
            dummy_me,
            param0=bayem.MVN([1, 1], np.diag([1, 1]), parameter_names=["A", "B"]),
        )

        dumped = json.dumps(result, cls=bayem.BayemEncoder, indent=2)

        loaded = json.loads(dumped, object_hook=bayem.bayem_hook)
        dumped_again = json.dumps(loaded, cls=bayem.BayemEncoder, indent=2)

        self.assertEqual(dumped, dumped_again)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import bayem
import json
from tempfile import TemporaryDirectory
from pathlib import Path


class TestJSON(unittest.TestCase):
    def test_json(self):
        data = {}
        data["parameter_prior"] = bayem.MVN(
            mean=np.r_[1, 2, 3],
            precision=np.diag([1, 2, 3]),
            parameter_names=["A", "B", "C"],
        )

        data["noise_prior"] = bayem.Gamma()
        data["non bayes thing"] = {"best number": 6174.0}

        string = bayem.save_json(data)
        loaded = bayem.load_json(string)

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

        with TemporaryDirectory() as f:
            filename = str(Path(f) / "tmp.json")
            dumped = bayem.save_json(result, filename)
            loaded = bayem.load_json(filename)
            dumped_again = bayem.save_json(loaded)
            self.assertEqual(dumped, dumped_again)



if __name__ == "__main__":
    unittest.main()

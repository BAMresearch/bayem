import unittest
import numpy as np
from bayes.vb import MVN


class TestMVN(unittest.TestCase):
    def setUp(self):
        self.mvn = MVN(
            mean=np.r_[1, 2, 3],
            precision=np.diag([1, 2, 3]),
            parameter_names=["A", "B", "C"],
        )

    def test_named_print(self):
        print(self.mvn)

    def test_named_access(self):
        self.assertEqual(self.mvn.index("A"), 0)
        self.assertEqual(self.mvn.named_mean("A"), 1)
        self.assertEqual(self.mvn.named_sd("A"), 1)

    def test_dim_mismatch(self):
        mean2 = np.random.random(2)
        prec2 = np.random.random((2, 2))
        prec3 = np.random.random((3, 3))
        with self.assertRaises(Exception):
            MVN(mean2, prec3)

        MVN(mean2, prec2)  # no exception!
        with self.assertRaises(Exception):
            MVN(mean2, prec2, parameter_names=["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()

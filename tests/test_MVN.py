import unittest
import numpy as np
import bayes.vb


class TestMVN(unittest.TestCase):
    def setUp(self):
        self.mvn = bayes.vb.MVN(
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
            bayes.vb.MVN(mean2, prec3)

        bayes.vb.MVN(mean2, prec2)  # no exception!
        with self.assertRaises(Exception):
            bayes.vb.MVN(mean2, prec2, parameter_names=["A", "B", "C"])

    def test_dist(self):
        mean = np.r_[1, 2, 3]
        prec = np.diag([1, 1 / 2 ** 2, 1 / 3 ** 2])

        mvn = bayes.vb.MVN(mean, prec)
        dist1D = mvn.dist(1)
        self.assertAlmostEqual(dist1D.mean(), 2)
        self.assertAlmostEqual(dist1D.std(), 2)


if __name__ == "__main__":
    unittest.main()

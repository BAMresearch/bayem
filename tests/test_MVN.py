import unittest
import numpy as np
import bayem


class TestMVN(unittest.TestCase):
    def setUp(self):
        self.mvn = bayem.MVN(
            mean=np.r_[1, 2, 3],
            precision=np.diag([1, 2, 3]),
            parameter_names=["A", "B", "C"],
        )

    def test_len(self):
        self.assertEqual(len(self.mvn), 3)

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
            bayem.MVN(mean2, prec3)

        bayem.MVN(mean2, prec2)  # no exception!
        with self.assertRaises(Exception):
            bayem.MVN(mean2, prec2, parameter_names=["A", "B", "C"])

    def test_dist(self):
        dist1D = self.mvn.dist(1)
        self.assertAlmostEqual(dist1D.mean(), 2)
        self.assertAlmostEqual(dist1D.std(), 1/2**0.5)

        dist2D = self.mvn.dist(1, 2)
        self.assertAlmostEqual(dist2D.mean[0], 2)
        self.assertAlmostEqual(dist2D.mean[1], 3)

        self.assertAlmostEqual(dist2D.cov[0, 0], 1/2)
        self.assertAlmostEqual(dist2D.cov[1, 1], 1/3)
        self.assertAlmostEqual(dist2D.cov[0, 1], 0)
        self.assertAlmostEqual(dist2D.cov[1, 0], 0)

        dist1D_named = self.mvn.dist("A")
        self.assertAlmostEqual(dist1D_named.mean(), 1)
        self.assertAlmostEqual(dist1D_named.std(), 1)


if __name__ == "__main__":
    unittest.main()

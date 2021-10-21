import unittest
import numpy as np
import bayes.vb
import scipy.stats
from hypothesis import given, settings
import hypothesis.strategies as st


class TestGamma(unittest.TestCase):
    def test_from_sd(self):
        gamma = bayes.vb.Gamma.FromSD(6174)
        self.assertAlmostEqual(gamma.mean, 1 / 6174 ** 2)

    def test_print(self):
        print(bayes.vb.Gamma.FromSD(42))

    def test_sd(self):
        scale, shape = 42, 6174
        gamma = bayes.vb.Gamma(shape=shape, scale=scale)
        self.assertAlmostEqual(gamma.mean, shape * scale)
        variance = shape * scale ** 2
        self.assertAlmostEqual(gamma.std, variance ** 0.5)

    def test_from_mean_and_sd(self):
        gamma = bayes.vb.Gamma.FromMeanStd(6174, 42)
        self.assertAlmostEqual(gamma.mean, 6174)
        self.assertAlmostEqual(gamma.std, 42)


    @settings(derandomize=True, max_examples=200)
    @given(st.tuples(st.floats(min_value=1.e-4, max_value=1e4), st.floats(min_value=1.e-4, max_value=1e4)))
    def test_from_quantiles(self, x0_x1):
        x0, x1 = x0_x1
        if x0 == x1:
            return
        if x0 > x1:
            x1, x0 = x0, x1

        q = (0.05, 0.95)
        gamma = bayes.vb.Gamma.FromQuantiles(x0, x1, q)
        d = gamma.dist()

        self.assertAlmostEqual(d.cdf(x0), q[0])
        self.assertAlmostEqual(d.cdf(x1), q[1])


if __name__ == "__main__":
    unittest.main()

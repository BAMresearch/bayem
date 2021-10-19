import unittest
import numpy as np
import bayes.vb


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


if __name__ == "__main__":
    unittest.main()

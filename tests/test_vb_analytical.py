import numpy as np
import unittest
import bayem

"""
Examples from
http://www.stat.cmu.edu/~brian/463-663/week09/Chapter%2003.pdf
section 3.4 about midterm results of 30 students.
"""

n = 30
np.random.seed(0)
data = np.random.normal(75, 10, size=n)
mx, sigma = np.mean(data), np.std(data)

prior_mean = 70
prior_sd = 5


def model_error(p):
    model = np.ones(n) * p[0]
    return model - data


class Test_VBAnalytic(unittest.TestCase):
    def setUp(self):
        """
        We infer the distribution mu for a given, fixed noise. This means no
        update of the gamma distribution within variational_bayes by passing
        the kwarg `update_noise={"error" : False}`

        For the prior N(prior_mean, scale=prior_sd), the parameters of the
        posterior distribution N(mean, scale) read
        """
        denom = n * prior_sd ** 2 + sigma ** 2
        self.mean = (sigma ** 2 * prior_mean + prior_sd ** 2 * n * mx) / denom
        variance = (sigma ** 2 * prior_sd ** 2) / denom
        self.scale = variance ** 0.5

    def given_noise(self, update_noise, check_method):
        prior = bayem.MVN(prior_mean, 1.0 / prior_sd ** 2)

        gamma = bayem.Gamma.FromMeanStd(1 / sigma ** 2, 42)

        result = bayem.variational_bayes(
            model_error, prior, gamma, update_noise=update_noise
        )
        self.assertTrue(result.success)
        check_method(result.param.mean[0], self.mean)
        check_method(result.param.std_diag[0], self.scale)

    def test_given_noise(self):
        self.given_noise(update_noise=False, check_method=self.assertAlmostEqual)
        self.given_noise(update_noise=True, check_method=self.assertNotAlmostEqual)

    def test_evidence(self):
        # analytic solution based on
        # https://www.econstor.eu/bitstream/10419/85883/1/02084.pdf eq 3 (whereever that is coming from)
        exponent = (
            -1
            / 2
            * (
                np.dot(data, data) / sigma ** 2
                + prior_mean ** 2 / prior_sd ** 2
                - self.mean ** 2 / self.scale ** 2
            )
        )

        z = (
            (2 * np.pi) ** (-n / 2)
            * sigma ** (-n)
            * self.scale
            / prior_sd
            * np.exp(exponent)
        )
        logz = np.log(z)

        prior = bayem.MVN(prior_mean, 1.0 / prior_sd ** 2)
        narrow_gamma = bayem.Gamma.FromMeanStd(1 / sigma ** 2, 1.0e-6)
        result1 = bayem.variational_bayes(
            model_error, prior, narrow_gamma, update_noise=True
        )
        result2 = bayem.variational_bayes(
            model_error, prior, narrow_gamma, update_noise=False
        )
        self.assertAlmostEqual(result1.f_max, logz, places=6)
        self.assertAlmostEqual(result2.f_max, logz, places=6)

    def test_given_mu(self):
        """
        We infer the gamma distribution of the noise precision for a
        given, fixed parameter mu. This is done by setting a prior with
        a very high precision.

        For a super noninformative noise prior (shape=0, scale->inf), the
        analytic solution for the INVERSE gamma distribution for the noise
        VARIANCE reads
        """
        a = n / 2
        b = np.sum((data - prior_mean) ** 2) / 2

        """
        The parameters for the corresponding gamma distribution for the 
        PRECISION then read (a, 1/b)
        """

        big_but_not_nan = 1e20
        prior = bayem.MVN(prior_mean, big_but_not_nan)
        gamma = bayem.Gamma(shape=1.0e-20, scale=big_but_not_nan)

        result = bayem.variational_bayes(
            model_error, prior, gamma, update_noise=True
        )
        self.assertTrue(result.success)

        gamma = result.noise
        self.assertAlmostEqual(gamma.shape, a)
        self.assertAlmostEqual(gamma.scale, 1 / b)

    def assert_gamma_equal(self, a, b):
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.scale, b.scale)

    def assert_not_gamma_equal(self, a, b):
        self.assertNotEqual(a.shape, b.shape)
        self.assertNotEqual(a.scale, b.scale)

    def test_inconsistent_noise(self):
        def dict_model_error(numbers):
            return {"A": np.ones(5), "B": np.zeros(5)}

        param0 = bayem.MVN()
        noise0 = {"A": bayem.Gamma(1, 1), "B": bayem.Gamma(2, 2)}

        # You may provide a single update_noise flag
        result = bayem.variational_bayes(
            dict_model_error, param0, noise0, update_noise=False
        )
        self.assert_gamma_equal(result.noise["A"], noise0["A"])
        self.assert_gamma_equal(result.noise["B"], noise0["B"])

        # Alternatively, you can provide a dict containing _all_ noise keys
        result = bayem.variational_bayes(
            dict_model_error, param0, noise0, update_noise={"A": True, "B": False}
        )
        self.assert_not_gamma_equal(result.noise["A"], noise0["A"])
        self.assert_gamma_equal(result.noise["B"], noise0["B"])

        # There will be an error, if you forget one
        with self.assertRaises(KeyError):
            bayem.variational_bayes(
                dict_model_error, param0, noise0, update_noise={"A": True}
            )


if __name__ == "__main__":
    unittest.main()

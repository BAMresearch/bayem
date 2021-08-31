import numpy as np
import unittest
import bayes.vb

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
    return {"error": model - data}

class Test_VBAnalytic(unittest.TestCase):
    def setUp(self):
        pass

    def test_given_noise(self):
        """
        We infer the distribution mu for a given, fixed noise. This means no 
        update of the gamma distribution within variational_bayes by passing
        the kwarg `update_noise={"error" : False}`

        For the prior N(prior_mean, scale=prior_sd), the parameters of the
        posterior distribution N(mean, scale) read
        """
        denom = n * prior_sd ** 2 + sigma ** 2
        mean = (sigma ** 2 * prior_mean + prior_sd ** 2 * n * mx) / denom
        variance = (sigma ** 2 * prior_sd ** 2) / denom
        scale = variance **0.5

        prior = bayes.vb.MVN(prior_mean, 1.0 / prior_sd ** 2)
        gamma = {"error": bayes.vb.Gamma.FromSD(sigma)}

        result = bayes.vb.variational_bayes(
            model_error, prior, gamma, update_noise={"error": False}
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.param.mean[0], mean)
        self.assertAlmostEqual(result.param.std_diag[0], scale)

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
        
        big_but_not_nan = 1e50
        prior = bayes.vb.MVN(prior_mean, big_but_not_nan)
        gamma = {"error": bayes.vb.Gamma(shape=0, scale=big_but_not_nan)}

        result = bayes.vb.variational_bayes(
            model_error, prior, gamma, update_noise={"error": True}
        )
        self.assertTrue(result.success)

        gamma = result.noise["error"]
        self.assertAlmostEqual(gamma.shape, a)
        self.assertAlmostEqual(gamma.scale, 1 / b)

if __name__ == "__main__":
    import logging

    # logging.basicConfig(level=logging.DEBUG)
    unittest.main()

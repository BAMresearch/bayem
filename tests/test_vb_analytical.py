import numpy as np
import unittest
import bayes.vb


class AnalyticProblem:
    def __init__(self):
        """
        Example from 
        http://www.stat.cmu.edu/~brian/463-663/week09/Chapter%2003.pdf
        section 3.4 about midterm results of 30 students.
        """

        # sample = student marks:
        self.n = 30
        self.data = np.random.normal(75, 10, size=self.n)
        self.mx = np.mean(self.data)
        self.sigma = np.std(self.data)

        # prior = results from previous classes
        self.prior_mean = 70
        self.prior_sd = 5

    def __call__(self, parameters):
        model = np.ones(self.n) * parameters[0]
        return {"error": model - self.data}

    def analytic_posterior(self, mx, sigma):
        tau = self.prior_sd
        denom = self.n * tau ** 2 + sigma ** 2
        mean = (sigma ** 2 * self.prior_mean + tau ** 2 * self.n * mx) / denom
        variance = (sigma ** 2 * tau ** 2) / denom
        return mean, variance


class Test_VBAnalytic(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_book_example(self):
        p = AnalyticProblem()
        mean, variance = p.analytic_posterior(mx=75, sigma=10)
        self.assertAlmostEqual(mean, 74.4, delta=0.1)
        print(variance)

    def run_test(self, update_noise, should_fail):
        np.random.seed(0)
        p = AnalyticProblem()
        prior = bayes.vb.MVN(p.prior_mean, 1.0 / p.prior_sd ** 2)
        gamma = {"error": bayes.vb.Gamma.FromSD(p.sigma)}

        result = bayes.vb.variational_bayes(p, prior, gamma, update_noise={"error":update_noise})
        self.assertTrue(result.success)

        mean, variance = p.analytic_posterior(mx=p.mx, sigma=p.sigma)

        if should_fail:
            compare = self.assertNotAlmostEqual
        else:
            compare = self.assertAlmostEqual

        compare(result.param.mean[0], mean)
        compare(result.param.std_diag[0], np.sqrt(variance))


    def test_vb(self):
        self.run_test(update_noise=False, should_fail=False)
        self.run_test(update_noise=True, should_fail=True)

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    unittest.main()


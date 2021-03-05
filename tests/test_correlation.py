import unittest
import numpy as np
import bayes.correlation
import bayes.vb


class TestCorrelation(unittest.TestCase):
    def test_fully_correlated(self):
        """corr should be almost ones"""
        x = np.linspace(0, 1, 10)
        corr = bayes.correlation.squared_exponential(x, 100)
        self.assertTrue(np.all(corr > 0.9))

        trans = bayes.correlation.transformation_from_correlation(corr)
        self.assertEqual(trans.shape[0], 10)
        self.assertLess(trans.shape[1], 5)

    def test_uncorrelated(self):
        """corr should be an identity matrix"""
        x = np.linspace(0, 1, 10)
        corr = bayes.correlation.squared_exponential(x, 0.02)
        self.assertLess(np.linalg.norm(corr - np.eye(10)), 1e-6)

        trans = bayes.correlation.transformation_from_correlation(corr)
        self.assertEqual(trans.shape[0], 10)
        self.assertEqual(trans.shape[1], 10)

    def test_invalid_2d(self):
        """corr should be an identity matrix"""
        x = np.eye(10)
        self.assertRaises(Exception, bayes.correlation.squared_exponential, x, 1.0)

    def test_with_MNV(self):
        np.random.seed(6174)
        x = np.linspace(0, 1, 100)
        corr = bayes.correlation.squared_exponential(x, 1e-1)

        values = np.random.multivariate_normal(np.zeros(len(x)), corr)
        trans = bayes.correlation.transformation_from_correlation(corr)
        # ???


class TestCorrelatedVB(unittest.TestCase):
    def run_vb(self, n_x, L_data, L_model):
        np.random.seed(6174)

        param_true = (7.0, 10.0)
        noise_std = 0.2
        x = np.linspace(0, 1, n_x)

        def fw(parameters):
            m, c = parameters
            return c + x * m

        perfect_data = fw(param_true)

        fine_grid_x = np.linspace(0, 1, 1000)
        noise_fine_grid = np.random.multivariate_normal(np.zeros(len(fine_grid_x)),
            bayes.correlation.squared_exponential(fine_grid_x, L_data) * noise_std ** 2, 1)[0]
        correlated_noise = np.interp(x, fine_grid_x, noise_fine_grid)

        data = perfect_data + correlated_noise

        transformation = bayes.correlation.transformation_SQ(x, L_model)

        def model_error(prm):
            return (fw(prm) - data) @ transformation

        param_prior = bayes.vb.MVN([6, 11], [[1 / 3 ** 2, 0], [0, 1 / 3 ** 2]])
        noise_prior = bayes.vb.Gamma(s=0.1, c=1000)

        result = bayes.vb.variational_bayes(model_error, param_prior, noise_prior)
        self.assertTrue(result.success)
        return result.param

    def test_demonstrate_issue(self):
        """More correlated data points makes the result more precise. Not good."""
        p50 = self.run_vb(50, L_data=0.5, L_model=1e-6)
        p200 = self.run_vb(200, L_data=0.5, L_model=1e-6)

        # The means are almost identical ...
        self.assertAlmostEqual(p200.mean[0], p50.mean[0], delta=1.0e-3*p200.mean[0])
        self.assertAlmostEqual(p200.mean[1], p50.mean[1], delta=1.0e-3*p200.mean[1])

        # ... but the standard deviation increases.
        self.assertLess(p200.std_diag[0], p50.std_diag[0] / 2.0)
        self.assertLess(p200.std_diag[1], p50.std_diag[1] / 2.0)

    def test_demonstrate_solution(self):
        """
        Including the correlation in the model 
        -- even though it is slightly off --
        makes the result independent from more data points
        """
        p50 = self.run_vb(50, L_data=0.5, L_model=0.5)
        p200 = self.run_vb(200, L_data=0.5, L_model=0.5)


        # Now, the standard deviations are almost identical ...
        self.assertAlmostEqual(
            p200.std_diag[0], p50.std_diag[0], delta=p50.std_diag[0] / 100
        )
        self.assertAlmostEqual(
            p200.std_diag[1], p50.std_diag[1], delta=p50.std_diag[1] / 100
        )
        # ... as well as the complete COV, ...
        diff_cov = np.abs(p50.cov - p200.cov)
        self.assertTrue(np.all(diff_cov < 1.e-3))

        # ... but the means are off... Is that expected?
        self.assertAlmostEqual(p200.mean[0], p50.mean[0], delta=1.0e-3*p200.mean[0])
        self.assertAlmostEqual(p200.mean[1], p50.mean[1], delta=1.0e-3*p200.mean[1])


if __name__ == "__main__":
    unittest.main()

from scipy import stats
import numpy as np
import unittest
import logging
from bayem import VBA, VBOptions

logging.getLogger("matplotlib.font_manager").disabled = True
logger = logging.getLogger(__name__)


def test_VBA(**options):
    return VBA(None, None, VBOptions(**options))


class TestFreeEnergy(unittest.TestCase):
    np.random.seed(12345)  # Seed the random number generator for reproducible results
    # Location, scale and weight for the two distributions
    dist1_loc, dist1_scale, weight1 = -1, 0.5, 0.25
    dist2_loc, dist2_scale, weight2 = 4, 1.75, 1.75
    dist3_loc, dist3_scale, weight3 = 9, 0.5, 0.40

    # Sample from a mixture of distributions
    param = np.linspace(-2, 12, 500)
    free_energy = (
        stats.norm.pdf(loc=dist1_loc, scale=dist1_scale, x=param) * weight1
        + stats.norm.pdf(loc=dist2_loc, scale=dist2_scale, x=param) * weight2
        + stats.norm.pdf(loc=dist3_loc, scale=dist3_scale, x=param) * weight3
    )

    r"""
                        __
                       /  \      __
         free energy  /    \    /  \
         ^           /      \__/    \
         |    __    /                \
         |   /  \__/                  \
         |  /                          \___
         |--------> param
          --------> iteration

    """

    def test_no_change_in_free_energy(self):
        """
        f_new - f_old < tol
        """
        msg = "---------Test case: Free energy is decreasing -max nb of trials was reached------- "
        logger.info(msg)
        # define samples of param and free energy for each iteration
        sample_interval = 10
        free_energy_sample = self.free_energy[::sample_interval]

        inference = test_VBA(
            maxtrials=10, tolerance=1e-4, maxiter=len(free_energy_sample)
        )
        for i_iter in range(inference.options.maxiter):
            f_new = free_energy_sample[i_iter]
            if inference.stop_criteria(f_new, i_iter):
                break

        print(
            inference.result.f_max, max(free_energy_sample), inference.options.tolerance
        )
        self.assertAlmostEqual(
            inference.result.f_max,
            max(free_energy_sample),
            delta=inference.options.tolerance,
            msg="iterations didn't go through the max in freee energy",
        )

    def test_increase_in_free_energy_at_max_iter(self):
        """
        f_new > f_old but i=maxiter
        :return:
        """
        msg = "---------Test case: Free energy is increasing - max nb of iterations is reached ---------"
        logger.info(msg)
        # define samples of param and free energy for each iteration
        sample_interval = 1
        free_energy_sample = self.free_energy[::sample_interval]

        inference = test_VBA(maxtrials=100, tolerance=1e-8, maxiter=50)
        for i_iter in range(inference.options.maxiter + 100):
            f_new = free_energy_sample[i_iter]
            if inference.stop_criteria(f_new, i_iter):
                break

        print(
            inference.result.f_max, max(free_energy_sample), inference.options.tolerance
        )
        self.assertLessEqual(
            inference.result.f_max,
            max(free_energy_sample),
            msg="result.f_max is cannot increase further",
        )
        self.assertEqual(inference.options.maxiter, i_iter)

    def test_stopped_by_max_iter(self):
        """
        f_new < f_old but i=maxiter and n_trials<maxtrials
        :return:
        """
        msg = "---------Test case: Free energy is decreasing -max nb of iterations is reached before maxtrials---------"
        logger.info(msg)
        # define samples of param and free energy for each iteration
        sample_interval = 1
        free_energy_sample = self.free_energy[::sample_interval]

        inference = test_VBA(maxtrials=100, tolerance=1e-5, maxiter=250)
        for i_iter in range(inference.options.maxiter):
            f_new = free_energy_sample[i_iter]
            if inference.stop_criteria(f_new, i_iter):
                break

        print(
            inference.result.f_max, max(free_energy_sample), inference.options.tolerance
        )
        self.assertAlmostEqual(
            inference.result.f_max,
            max(free_energy_sample),
            delta=inference.options.tolerance,
            msg="iterations didn't go through the max in freee energy",
        )
        self.assertLessEqual(
            inference.n_trials, inference.options.maxtrials, msg="maxtrials was reached"
        )

    def test_reached_tolerance_return_stored(self):
        """
        f_new < f_old but i=maxiter and n_trials<maxtrials
        :return:
        """
        msg = "---------Test case: Free energy change is below tolerance - return stored param values---------T"
        logger.info(msg)
        # define samples of param and free energy for each iteration
        sample_interval = 10
        free_energy_sample = self.free_energy[::sample_interval]

        inference = test_VBA(
            maxtrials=100, tolerance=5e-3, maxiter=len(free_energy_sample)
        )
        for i_iter in range(inference.options.maxiter):
            f_new = free_energy_sample[i_iter]
            if inference.stop_criteria(f_new, i_iter):
                break

        print(
            inference.result.f_max, max(free_energy_sample), inference.options.tolerance
        )
        self.assertAlmostEqual(
            inference.f_old,
            f_new,
            delta=inference.options.tolerance,
            msg="change in free energy is higher than tolerance",
        )
        self.assertAlmostEqual(
            inference.result.f_max,
            max(free_energy_sample),
            delta=inference.options.tolerance,
            msg="result.f_max didnt reach max free energy ",
        )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

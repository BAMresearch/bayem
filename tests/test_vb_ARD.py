import numpy as np
import unittest
import bayem
from test_vb import ModelError, ModelErrorWithJacobian


class ForwardModel:
    def __init__(self):
        self.xs = np.linspace(0.0, 10, 20)

    def __call__(self, parameters):
        m = parameters[0]
        c = parameters[1]
        b = parameters[2:]
        return c + self.xs * m + b

    def jacobian(self, parameters):
        b = parameters[2:]

        d_dm = self.xs
        d_dc = np.ones_like(self.xs)
        all_d = [d_dm, d_dc]
        for i in b:
            all_d.append([1 for x in self.xs])

        return np.array(all_d).T


class Test_VB(unittest.TestCase):
    """
    Test having an ARD parameter (see Chappell paper for more info)
    True model (used for data generation) is defined as a linear model + bias term at each "sensor" location (xs)
    Bias term represents how the fw model deviates from the true model and should be inferred during VB. Since the bias parameters are sparse, an ARD prior is set for them.
    """

    def run_vb(self, n_data, given_jac=False):
        np.random.seed(6174)

        fw = ForwardModel()
        n_sensors = len(fw.xs)
        param_true = [7.0, 10.0]

        bias_true = [0] * (n_sensors)
        bias_true[3] = 5
        param_true = param_true + bias_true
        #        noise_std =0.1
        noise_std = 0.01 * abs(np.mean(fw(param_true)))

        data = []
        perfect_data = fw(param_true)
        for _ in range(n_data):
            data.append(
                perfect_data + np.random.normal(0, noise_std, len(perfect_data))
            )

        if given_jac:
            me = ModelErrorWithJacobian(fw, data)
        else:
            me = ModelError(fw, data)

        # setting mean and precision
        bias_param = [[0] * n_sensors, [1e-3] * n_sensors]
        param_prior = bayem.MVN(
            [6, 11] + bias_param[0],
            ([1 / 3 ** 2, 1 / 3 ** 2] + bias_param[1]) * np.identity(2 + n_sensors),
        )
        noise_prior = bayem.Gamma(scale=0.5, shape=2 * 1 / noise_std ** 2)

        vb = bayem.VB()
        info = vb.run(
            me,
            param_prior,
            noise_prior,
            index_ARD=np.arange(2, n_sensors + 2),
            iter_max=100,
            n_trials_max=50,
        )
        print(info)
        param_post, noise_post = info.param, info.noise

        for i, p in enumerate(param_true):
            if i < 2 or abs(p) > 1e-10:
                posterior_mean = param_post.mean[i]
                posterior_std = param_post.std_diag[i]
                print(
                    "Param {} True value = {} ".format(i, param_true[i]),
                    "\n inferred = {} +- {}".format(posterior_mean, posterior_std),
                )
                # self.assertLess(posterior_std, 0.3)
                self.assertAlmostEqual(
                    posterior_mean, param_true[i], delta=2 * posterior_std
                )

        post_noise_precision = noise_post.mean
        post_noise_std = 1.0 / post_noise_precision ** 0.5
        self.assertAlmostEqual(post_noise_std, noise_std, delta=noise_std / 100)

    def test_vb_with_given_jac(self):
        self.run_vb(n_data=10, given_jac=False)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

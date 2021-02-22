import numpy as np
import unittest
from bayes.vb import *

class ForwardModel:
    def __init__(self):
        self.xs = np.linspace(.01, .1, 10)

    def __call__(self, parameters):
        m, c = parameters
        v = [c + x * m for x in self.xs]
        return np.array(v)

    def jacobian(self, parameters):
        m, c = parameters
        d_dm = [x for x in self.xs]
        d_dc = [1 for x in self.xs]
        return np.array([d_dm, d_dc]).T


def stack_data(model_response, n_data, noise_std = None):
    """
    Creates a big vector by stacking `model_response` `n_data` times.

    If noise_std is not none, it adds a normal distribution with
    mean=0 and std=noise_std. This can be used to create noisy data.

    The idea: Use the same method for both stacking the model response _and_
    creating the data, such that the ordering cannot be messed up.
    """
    result = np.repeat(model_response, n_data, axis=0)

    if noise_std is not None:
        result += np.random.normal(0, noise_std, len(result))

    return result



class ModelError:
    def __init__(self, forward_model, data):
        """
        forward_model:
            forward model
        data:
            positions to evaluate, could correspond to sensor positions
        """
        self._forward_model = forward_model
        self._data = data

        if not hasattr(forward_model, "jacobian"):
            delattr(self, "jacobian")

    def __call__(self, parameters):
        model = self._forward_model(parameters)
        repetitions = self._data.shape[0] // model.shape[0]

        return self._data - stack_data(model, repetitions)

class ModelErrorWithJacobian(ModelError): 
    def jacobian(self, parameters):
        jac = self._forward_model.jacobian(parameters)
        repetitions = self._data.shape[0] // jac.shape[0]
        return stack_data(jac, repetitions)


class Test_VB(unittest.TestCase):

    def run_vb(self, n_data, given_jac=False, plot=False):
        np.random.seed(6174)

        fw = ForwardModel()
        param_true = (7., 10.)
        noise_std = 0.1

        data = stack_data(fw(param_true), n_data, noise_std)

        if given_jac:
            me = ModelErrorWithJacobian(fw, data)
        else:
            me = ModelError(fw, data)

        param_prior = MVN([6, 11], [[1 / 3 ** 2, 0], [0, 1 / 3 ** 2]])
        noise_prior = Gamma(s=0.1, c=1000)

        info = variational_bayes(me, param_prior, noise_prior)
        param_post, noise_post = info.param, info.noise

        if plot:
            plot_pdf(param_post, expected_value=param_true, compare_with=param_prior, plot="joint")
       
        for i in range(2):
            posterior_mean = param_post.mean[i]
            posterior_std = param_post.std_diag[i]

            self.assertLess(posterior_std, 0.3)
            self.assertAlmostEqual(posterior_mean, param_true[i], delta=2 * posterior_std)
            
        post_noise_precision = noise_post[0].mean
        post_noise_std = 1. / post_noise_precision**0.5
        self.assertAlmostEqual(post_noise_std, noise_std, delta=noise_std/100)
        
        self.assertLess(info.nit, 20)
        print(info)

    def test_vb_with_numeric_jac(self):
        self.run_vb(n_data=1000, given_jac=False)

    def test_vb_with_given_jac(self):
        self.run_vb(n_data=1000, given_jac=True, plot=False)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

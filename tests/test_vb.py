import numpy as np
import pytest
import bayem


class ForwardModel:
    def __init__(self):
        self.xs = np.linspace(0.01, 0.1, 10)

    def __call__(self, parameters):
        m, c = parameters
        return c + self.xs * m

    def jacobian(self, parameters):
        m, c = parameters
        d_dm = self.xs
        d_dc = np.ones_like(self.xs)
        return np.vstack([d_dm, d_dc]).T


class ModelError:
    def __init__(self, forward_model, data, with_jacobian):
        """
        forward_model:
            forward model
        data:
            positions to evaluate, could correspond to sensor positions
        """
        self._forward_model = forward_model
        self._data = data
        self.with_jacobian = with_jacobian

    def __call__(self, parameters):
        model = self._forward_model(parameters)
        errors = []
        for data in self._data:
            errors.append(model - data)

        k = np.concatenate(errors)
        if self.with_jacobian:
            jac = self._forward_model.jacobian(parameters)
            return k, np.tile(jac, (len(self._data), 1))
        else:
            return k

@pytest.mark.parametrize("n_data, given_jac", [(1000, False), (1000, True)])
def test_vb(n_data, given_jac):
    np.random.seed(6174)

    fw = ForwardModel()
    param_true = [7.0, 10.0]
    noise_sd = 0.1

    data = []
    perfect_data = fw(param_true)
    for _ in range(n_data):
        data.append(perfect_data + np.random.normal(0, noise_sd, len(perfect_data)))

    me = ModelError(fw, data, given_jac)

    param_prior = bayem.MVN([0, 11], [[1 / 7 ** 2, 0], [0, 1 / 3 ** 2]])
    noise_prior = bayem.Gamma.FromSDQuantiles(0.5 * noise_sd, 1.5 * noise_sd)

    info = bayem.vba(me, param_prior, noise_prior, jac=given_jac)

    param_post, noise_post = info.param, info.noise

    for i in range(2):
        posterior_mean = param_post.mean[i]
        posterior_std = param_post.std_diag[i]

        assert posterior_std < 0.3
        assert posterior_mean == pytest.approx(param_true[i], abs=2 * posterior_std)

        post_noise_precision = noise_post.mean
    post_noise_sd = 1.0 / post_noise_precision ** 0.5
    assert post_noise_sd == pytest.approx(noise_sd, rel=0.01)

    assert info.nit < 20
    print(info)


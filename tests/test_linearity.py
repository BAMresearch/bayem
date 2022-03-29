import numpy as np
import pytest

import bayem

N_sensors = 20
x = np.linspace(0, 1, N_sensors)


def test_linear_model():
    def f(theta):
        return {"noise_group": x * theta[0] + theta[1]}

    mvn = bayem.MVN.FromMeanStd([1, 1], [0.5, 0.5], parameter_names=["A", "B"])

    sd_range = range(-3, 4)
    result, values = bayem.linearity_analysis(f, mvn, sd_range=sd_range, ret_values=True)
    assert 0 < result["noise_group"]["A"] < 1.0e-8
    assert 0 < result["noise_group"]["B"] < 1.0e-8

    # test access to raw values
    test_me_true, test_me_lin = values["noise_group"]["A"]
    assert len(test_me_true) == len(sd_range)
    for me in test_me_true:
        assert len(me) == N_sensors


def test_custom_sd_range():
    def f(theta):
        return {"noise_group": x / theta[0]}

    # The default setting will test a parameter range
    # -2, -1, 0, 1, 2, 3, 4
    # where theta == 0 causes a division by zero in the model.
    with pytest.warns(RuntimeWarning):
        bayem.linearity_analysis(f, bayem.MVN(1, 1))

    # We can avoid that by manually adjusting the checked range.
    bayem.linearity_analysis(f, bayem.MVN(1, 1), sd_range=np.linspace(-0.9, 0.9, 5))


def test_nonlinear_model():
    def f(theta):
        return x / theta[0] + theta[1]

    # MVN with its mean in region of high nonlinearity
    mvn0 = bayem.MVN.FromMeanStd([1, 1], [0.3, 0.3], parameter_names=[0, 1])

    result0 = bayem.linearity_analysis(f, mvn0)
    assert result0[0] > 0.1
    assert 0 < result0[1] < 1.0e-8  # the model is still linear in theta[1]

    # MVN with its mean in region of _low_ nonlinearity
    mvn1 = bayem.MVN.FromMeanStd([10, 1], [0.2, 0.2], parameter_names=[0, 1])
    result1 = bayem.linearity_analysis(f, mvn1)
    assert 0 < result1[0] < 0.001  # lower "measure" of nonlinearity
    assert 0 < result1[1] < 1.0e-8  # the model is still linear in theta[1]


def test_zero_model():
    def f(theta):
        """
        Could triggers division by zero in norm calculation
        """
        return np.zeros(42)

    a = bayem.linearity_analysis(f, bayem.MVN(4, 2, parameter_names=["A"]))
    assert np.isnan(a["A"])

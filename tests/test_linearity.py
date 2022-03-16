import numpy as np
import bayem
import pytest

N_sensors = 20
x = np.linspace(0, 1, N_sensors)


def test_linear_model():
    def f(theta):
        return x * theta[0] + theta[1]

    mvn = bayem.MVN.FromMeanStd([1, 1], [0.5, 0.5], parameter_names=[0, 1])

    result = bayem.linearity_analysis(f, mvn)
    assert 0 < result[0] < 1.0e-8
    assert 0 < result[1] < 1.0e-8


def test_nonlinear_model():
    def f(theta):
        return x / theta[0] + theta[1]

    # MVN with its mean in region of high nonlinearity
    mvn0 = bayem.MVN.FromMeanStd([1, 1], [0.3, 0.3], parameter_names=[0, 1])

    result0 = bayem.linearity_analysis(f, mvn0, show=True)
    assert result0[0] > 0.1
    assert 0 < result0[1] < 1.0e-8 # the model is still linear in theta[1]

    # MVN with its mean in region of _low_ nonlinearity
    mvn1 = bayem.MVN.FromMeanStd([10, 1], [0.2, 0.2], parameter_names=[0, 1])
    result1 = bayem.linearity_analysis(f, mvn1)
    assert 0 < result1[0] < 0.001  # lower "measure" of nonlinearity
    assert 0 < result1[1] < 1.0e-8 # the model is still linear in theta[1]

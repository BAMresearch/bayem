import hypothesis.strategies as st
import pytest
from hypothesis import given, settings, assume

import bayem


def test_print():
    print(bayem.Gamma(42, 2))


def test_sd():
    scale, shape = 42, 6174
    gamma = bayem.Gamma(shape=shape, scale=scale)

    assert gamma.mean == pytest.approx(shape * scale)
    assert gamma.std ** 2 == pytest.approx(shape * scale ** 2)  # variance


@settings(derandomize=True, max_examples=200)
@given(
    st.tuples(
        st.floats(min_value=1.0e-4, max_value=1e4),
        st.floats(min_value=1.0e-4, max_value=1e4),
    )
)
def test_from_quantiles(x0_x1):
    x0, x1 = x0_x1
    assume(abs(x1 - x0) > 1e-6)
    assume(x0 < x1)

    q = (0.15, 0.95)
    gamma = bayem.Gamma.FromQuantiles(x0, x1, q)
    d = gamma.dist()

    assert d.cdf(x0) == pytest.approx(q[0])
    assert d.cdf(x1) == pytest.approx(q[1])


def test_from_mean_and_sd():
    gamma_ref = bayem.Gamma(6174, 42)

    gamma = bayem.Gamma.FromMeanStd(gamma_ref.mean, gamma_ref.std)
    assert gamma == gamma_ref


def test_from_sd_quantiles():
    gamma = bayem.Gamma.FromSDQuantiles(4, 6)
    sd_mean = 1 / gamma.mean ** 0.5

    assert 4 < sd_mean < 6

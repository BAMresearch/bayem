import numpy as np
import pytest

import bayem

mvn = bayem.MVN(
    mean=np.r_[1, 2, 3],
    precision=np.diag([1, 2, 3]),
    parameter_names=["A", "B", "C"],
)


def test_len():
    assert len(mvn) == 3


def test_named_print():
    print(mvn)


def test_named_access():
    assert mvn.index("A") == 0
    assert mvn.named_mean("A") == 1
    assert mvn.named_sd("A") == 1


def test_dim_mismatch():
    mean2 = np.random.random(2)
    prec2 = np.random.random((2, 2))
    prec3 = np.random.random((3, 3))
    with pytest.raises(Exception):
        bayem.MVN(mean2, prec3)

    bayem.MVN(mean2, prec2)  # no exception!
    with pytest.raises(Exception):
        bayem.MVN(mean2, prec2, parameter_names=["A", "B", "C"])


def test_dist():
    dist1D = mvn.dist(1)
    assert dist1D.mean() == 2
    assert dist1D.std() == pytest.approx(1 / 2 ** 0.5)

    dist2D = mvn.dist(1, 2)
    assert dist2D.mean[0] == 2
    assert dist2D.mean[1] == 3

    assert dist2D.cov[0, 0] == 1 / 2
    assert dist2D.cov[1, 1] == 1 / 3
    assert dist2D.cov[0, 1] == 0
    assert dist2D.cov[1, 0] == 0

    dist1D_named = mvn.dist("A")
    assert dist1D_named.mean() == 1
    assert dist1D_named.std() == 1


def test_simple_init():
    # this mvn ...
    mvn_ref = bayem.MVN(
        [1, 2],
        np.diag([1 / 2 ** 2, 1 / 3 ** 2]),
        name="test",
        parameter_names=["A", "B"],
    )
    # ... can be directly created via
    mvn_simple = bayem.MVN.FromMeanStd(
        [1, 2], [2, 3], name="test", parameter_names=["A", "B"]
    )
    assert mvn_ref == mvn_simple

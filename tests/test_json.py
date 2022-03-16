from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import bayem


def test_result_io():
    def dummy_me(prm):
        return prm ** 2

    result = bayem.vba(
        dummy_me,
        x0=bayem.MVN([1, 1], np.diag([1, 1]), parameter_names=["A", "B"]),
    )

    with TemporaryDirectory() as f:
        filename = str(Path(f) / "tmp.json")
        dumped = bayem.save_json(result, filename)
        loaded = bayem.load_json(filename)
        dumped_again = bayem.save_json(loaded)
        assert dumped == dumped_again


def test_custom_objects():
    data = {}
    data["parameter_prior"] = bayem.MVN(
        mean=np.r_[1, 2, 3],
        precision=np.diag([1, 2, 3]),
        parameter_names=["A", "B", "C"],
    )

    data["noise_prior"] = bayem.Gamma()
    data["non bayes thing"] = {"best number": 6174.0}

    string = bayem.save_json(data)
    loaded = bayem.load_json(string)

    A, B = data["parameter_prior"], loaded["parameter_prior"]
    CHECK = np.testing.assert_array_equal
    assert A.name == B.name
    assert A.parameter_names == B.parameter_names
    CHECK(A.mean, B.mean)
    CHECK(A.precision, B.precision)

    C, D = data["noise_prior"], loaded["noise_prior"]
    assert C.name == D.name
    assert C.shape == D.shape
    assert C.scale == D.scale

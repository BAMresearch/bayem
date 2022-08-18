import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from imageio import imread

import bayem
import bayem.visualization as visu


def compare_plt(reference_filename, generate_ref_img):
    ref_img_name = Path(__file__).absolute().parent / reference_filename
    if generate_ref_img:
        plt.savefig(ref_img_name, dpi=300)
        return

    with tempfile.TemporaryDirectory() as tmpdirname:
        test_img_name = Path(tmpdirname) / "test_visu.png"
        plt.savefig(test_img_name, dpi=300)

        test_img = imread(test_img_name)
        ref_img = imread(ref_img_name)

        assert np.linalg.norm(test_img - ref_img) == pytest.approx(0)


np.random.seed(6174)
t = np.linspace(1, 2, 10)
noise = np.random.normal(0, 0.42, len(t))


def f(x):
    return t * x[0] ** 2 - t * 9 + noise


info = bayem.vba(f, x0=bayem.MVN([2], [0.5]), noise0=bayem.Gamma(1, 2))


def test_pair_plot(generate_ref_img=False):
    # use crafted posterior close to the prior for visually more interesting
    # plot
    info.param = bayem.MVN(1.5, 0.3)
    visu.pair_plot(info, show=False)
    compare_plt("ref_pair_plot.png", generate_ref_img=generate_ref_img)


def test_trace_plot(generate_ref_img=False):
    visu.result_trace(info, show=False)
    compare_plt("ref_trace_plot.png", generate_ref_img=generate_ref_img)


def test_quantile_levels():
    mvn = bayem.MVN([2, 3], [[0.5, 0.2], [0.2, 0.5]])
    res = 201
    x0 = np.linspace(mvn.dist(0).ppf(0.001), mvn.dist(0).ppf(0.999), res)
    x1 = np.linspace(mvn.dist(1).ppf(0.001), mvn.dist(1).ppf(0.999), res)
    xi, xj = np.meshgrid(x0, x1)

    x = np.vstack([xi.flatten(), xj.flatten()]).T
    pdf = mvn.dist(0, 1).pdf(x).reshape(res, res)

    alpha = 0.9
    level = visu._get_levels(pdf, alpha)

    level_ref = alpha * (1 / np.sqrt((2 * np.pi) ** 2) / np.linalg.det(mvn.cov)) * 0.9

    assert level_ref == pytest.approx(level)


if __name__ == "__main__":
    generate = True
    test_quantile_levels()
    test_pair_plot(generate_ref_img=generate)
    test_trace_plot(generate_ref_img=generate)

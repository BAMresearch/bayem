import pytest
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import bayem
from imageio import imread


def test_vb_visu(generate_ref_img=False):
    mvn = bayem.MVN(mean=[2, 30, 100], precision=[[1, 1, 1], [1, 2, 0], [1, 0, 3]])
    gamma0 = bayem.Gamma(shape=3, scale=1 / 75)
    gamma1 = bayem.Gamma(shape=8.5, scale=1 / 8.5)

    axes = bayem.visualize_vb_marginal_matrix(mvn, [gamma0, gamma1], label="VB")
    bayem.format_axes(axes)

    ref_img_name = Path(__file__).absolute().parent / "test_vb_visu_ref.png"
    if generate_ref_img:
        plt.savefig(ref_img_name, dpi=300)
        return

    with tempfile.TemporaryDirectory() as tmpdirname:
        test_img_name = Path(tmpdirname) / "test_visu.png"
        plt.savefig(test_img_name, dpi=300)

        test_img = imread(test_img_name)
        ref_img = imread(ref_img_name)

        assert np.linalg.norm(test_img - ref_img) == pytest.approx(0)


if __name__ == "__main__":
    test_vb_visu(generate_ref_img=False)

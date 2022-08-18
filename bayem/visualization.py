from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d

from bayem import Gamma


def plot_pdf(
    dist,
    expected_value=None,
    name1="Posterior",
    name2="Prior",
    plot="individual",
    color_plot="r",
    **kwargs,
):
    if ("min" in kwargs) & ("max" in kwargs):
        x_min = kwargs["min"]
        x_max = kwargs["max"]
    else:
        x_min = dist.ppf(0.001)
        x_max = dist.ppf(0.999)

    x_plot = np.linspace(x_min, x_max, 5000)

    if expected_value is not None:
        expected_value = np.atleast_1d(expected_value)

    for i in range(len(dist.mean)):
        plt.plot(x_plot, dist.pdf(x_plot), label="%s of parameter nb %i" % (name1))
        if expected_value is not None:
            plt.axvline(expected_value[i], ls=":", red="k")
        if "compare_with" in kwargs:
            plt.plot(
                x_plot,
                kwargs["compare_with"].pdf(x_plot),
                label="%s of parameter nb %i" % (name2),
            )

        plt.xlabel("Parameter")
        plt.legend()
        if plot == "individual":
            plt.show()
    if plot == "joint":
        plt.show()


def visualize_vb_marginal_matrix(
    mvn,
    gammas=None,
    axes=None,
    resolution=201,
    median=True,
    color="#d20020",
    lw=1,
    label=None,
    legend_fontsize=8,
    focus=False,
    contour_quantiles=np.r_[0.5, 0.7, 0.9],
):
    """
    Creates a plot grid with the analytical marginal plots of `mvn` and
    `gammas` or adds those to an existing one.

    mvn:
        Any multivariate normal distribution that provides `.mean` and `.cov`
    gammas:
        A list of `bayem.Gamma` distributions
    axes:
        If provided, this is 2D array containing similar plots e.g. from
            taralli_model.plot_posterior(...)
        or
            arviz.plot_pair(...)
    resolution:
        Number of points in the range of ppf(0.001) .. ppf(0.999) used for the
        visualization. More points result in smoother lines.
    median:
        If true, also shows median lines
    color:
        Color that is used for all lines. All matplotlib compatible formats
        should work like "red", "g", (0.61, 0.47, 0)
    lw:
        widths of all plotted lines
    label:
        if provided, adds a legend entry
    focus:
        if true, adjusts the axis limits to the current data
    """
    gammas = gammas or []

    N_mvn = len(mvn.mean)
    N = N_mvn + len(gammas)

    axes = _get_axes(N, axes)

    dists_1d = [mvn.dist(i) for i in range(N_mvn)] + [gamma.dist() for gamma in gammas]

    xs = []
    for i in range(N):
        xs.append(
            np.linspace(dists_1d[i].ppf(0.001), dists_1d[i].ppf(0.999), resolution)
        )

        # diagonal plots: 1D line plots of the pdfs
        x = xs[i]
        axes[i, i].plot(x, dists_1d[i].pdf(x), "-", color=color, lw=lw)

        for j in range(i):
            # off-diagonal plots: 2D contour plots of the pdfs
            xi, xj = np.meshgrid(xs[i], xs[j])

            if i < N_mvn and j < N_mvn:
                # correlated pdf by evaluating 2D-mvn
                dist = mvn.dist(i, j)
                x = np.vstack([xi.flatten(), xj.flatten()]).T
                pdf = dist.pdf(x).reshape(resolution, resolution)
            else:
                # uncorrelated pdf by multiplying the individual pdfs
                pdf_i = dists_1d[i].pdf(xs[i]).reshape(-1, 1)
                pdf_j = dists_1d[j].pdf(xs[j]).reshape(-1, 1)
                pdf = pdf_j @ pdf_i.T

                def get_levels(quantiles):
                    level_range = np.linspace(np.min(pdf), np.max(pdf), 15)[:-1]

                    sums = [np.sum(pdf[pdf > level]) for level in level_range]
                    prob_density = np.asarray(sums) / np.sum(pdf)

                    interpolator = interp1d(prob_density, level_range, kind="quadratic")
                    return interpolator(quantiles)

                levels = get_levels(np.sort(contour_quantiles)[::-1])
                axes[i, j].contour(
                    xj, xi, pdf, levels=levels, colors=[color], linewidths=lw
                )

    if median:
        for i in range(N):
            axes[i, i].axvline(dists_1d[i].median(), ls="--", color=color, lw=lw)
            for j in range(i):
                axes[i, j].axhline(dists_1d[i].median(), ls="--", color=color, lw=lw)
                axes[i, j].axvline(dists_1d[j].median(), ls="--", color=color, lw=lw)

    if focus:
        _set_focus(axes, xs)

    if label is None:
        return axes

    line = Line2D([0], [0], color=color, linewidth=lw, label=label)
    fig = axes[0, 0].figure
    if fig.legends:
        handles = fig.legends[0].legendHandles
        fig.legends = []
    else:
        handles = []
    handles.append(line)

    ax_bounds = np.array([list(ax.bbox._bbox.bounds) for ax in axes.flat])
    top_margin = 1 - (ax_bounds[:, 0] + ax_bounds[:, 2]).max()
    right_margin = 1 - (ax_bounds[:, 1] + ax_bounds[:, 3]).max()
    bbox_to_anchor = (1.0 - right_margin / 2, 1.0 - top_margin)
    fig.legend(
        handles=handles,
        loc="upper right",
        fontsize=legend_fontsize,
        bbox_to_anchor=bbox_to_anchor,
    )

    return axes


def _get_axes(N, axes):
    if axes is None:
        fig = plt.figure()
        axes = fig.subplots(N, N, sharex="col", squeeze=False)
    assert axes.shape == (N, N)
    return axes


def _set_focus(axes, xs):
    assert len(axes) == len(xs)
    N = len(axes)
    for i in range(N):
        axes[i, i].set_xlim(xs[i][0], xs[i][-1])
        for j in range(i + 1):
            axes[i, j].set_ylim(xs[i][0], xs[i][-1])


def format_axes(axes, labels=None):
    """
    See the comments below for the individual adjustments made.

    axes:
        2D matplotlib.axes grid containing the marginal matrix
    labels:
        names of the individual variables
    """
    N = len(axes)

    # add labels
    labels = labels or [fr"$\theta_{i}$" for i in range(N)]
    assert N == len(labels)
    for i, label in enumerate(labels):
        axes[i, i].set_ylabel(f"$PDF$({labels[i]})")
        axes[len(labels) - 1, i].set_xlabel(label)

    # turn off top triangle
    for i in range(N):
        for j in range(i + 1, N):
            axes[i, j].axis("off")

    # move all y tick labels to the very right plot of the row
    for i in range(N):
        axes[i, i].yaxis.set_label_position("right")
        axes[i, i].yaxis.tick_right()
        for j in range(0, i):
            axes[i, j].yaxis.set_ticks([])


class PairPlot:
    """
    Class to conveniently define pair plots for a VB analysis with some
    reasonable default values.
    """

    def __init__(self, result):
        self.axes = None
        self.result = result
        noise_keys = [r"$\phi_{" + str(key) + "}$" for key in result.noise0_dict]
        self.labels = result.param0.parameter_names + noise_keys

    def prior(self, **kwargs):
        kwargs.setdefault("color", "#cd7e00")
        kwargs.setdefault("lw", 0.5)
        kwargs.setdefault("label", "prior")
        self.axes = visualize_vb_marginal_matrix(
            self.result.param0,
            self.result.noise0_dict.values(),
            axes=self.axes,
            **kwargs,
        )
        return self

    def posterior(self, **kwargs):
        kwargs.setdefault("color", "#d2001e")
        kwargs.setdefault("lw", 1.5)
        kwargs.setdefault("label", "vb posterior")
        self.axes = visualize_vb_marginal_matrix(
            self.result.param, self.result.noise_dict.values(), axes=self.axes, **kwargs
        )
        return self

    def finish(self, show=True):
        format_axes(self.axes, self.labels)
        if show:
            plt.show()


def pair_plot(result, show=True):
    PairPlot(result).prior().posterior().finish(show)


def result_trace(result, show=True, highlight=None):
    if highlight is None:
        highlight = []

    fig, (ax_f, ax_p, ax_g) = plt.subplots(3, 1, sharex=True)

    # plot parameters
    ax_p.set_xlabel("interation")
    ax_p.set_ylabel("parameter value")
    ax_g.set_ylabel("noise precision")
    ax_p.xaxis.set_major_locator(MaxNLocator(integer=True))

    means = np.array([result.param0.mean] + result.means)
    sds = np.array([result.param0.std_diag] + result.sds)
    x = list(range(len(means)))

    plt.subplots_adjust(wspace=0, hspace=0)

    red = np.r_[210, 0, 30] / 255

    for i, name in enumerate(result.param.parameter_names):
        color, lw = None, 1
        if name in highlight:
            color, lw = red, 2
        ax_p.errorbar(
            x, means[:, i], yerr=sds[:, i], label=name, capsize=5, color=color, lw=lw
        )

    ax_p.legend()

    # plot free energy
    ax_f.set_ylabel("free energy")
    ax_f.plot(x[1:], result.free_energies, "-k|")

    for noise_key, gammas in result.gammas.items():
        g0 = result.noise0_dict[noise_key]
        means = np.array([g0.mean] + [g.mean for g in gammas])
        sds = np.array([g0.std] + [g.std for g in gammas])
        label = r"$\phi_{" + str(noise_key) + "}$"
        ax_g.errorbar(x, means, yerr=sds, label=label, capsize=5, lw=1, ls=":")
    ax_g.legend()

    # annotate prior and posterior
    i_posterior = result.free_energies.index(result.f_max) + 1
    ax_f.axvline(i_posterior, color=red)
    ax_f.axvline(0, color="orange")
    y = np.min(result.free_energies)
    ax_f.text(0, y, "prior", color="orange")
    ax_f.text(i_posterior, y, "posterior", color=red)

    if show:
        plt.show()

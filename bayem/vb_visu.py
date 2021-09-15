import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_pdf(
    dist,
    expected_value=None,
    name1="Posterior",
    name2="Prior",
    plot="individual",
    color_plot="r",
    **kwargs,
):
    import matplotlib.pyplot as plt

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
            plt.axvline(expected_value[i], ls=":", c="k")
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
    legend_fontsize=8
):
    """
    Creates a plot grid with the analytical marginal plots of `mvn` and
    `gammas` or adds those to an existing one.

    mvn:
        Any multivariate normal distribution that provides `.mean` and `.cov`
    gammas:
        A list of `bayes.vb.Gamma` distributions
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
    """
    gammas = gammas or []

    N_mvn = len(mvn.mean)
    N = N_mvn + len(gammas)

    if axes is None:
        _, axes = plt.subplots(N, N)

    assert axes.shape == (N, N)

    dists_1d = [mvn.dist(i) for i in range(N_mvn)] + [gamma.dist() for gamma in gammas]

    xs = []
    for i in range(N):
        xs.append(
            np.linspace(dists_1d[i].ppf(0.001), dists_1d[i].ppf(0.999), resolution)
        )

        for j in range(i + 1):
            if i == j:
                x = xs[i]
                axes[i, i].plot(x, dists_1d[i].pdf(x), "-", color=color, lw=lw)
                if median:
                    axes[i, i].axvline(
                        dists_1d[i].median(), ls="--", color=color, lw=lw
                    )
            else:
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

                axes[i, j].contour(xj, xi, pdf, colors=[color], linewidths=lw)
                if median:
                    axes[i, j].axhline(
                        dists_1d[i].median(), ls="--", color=color, lw=lw
                    )
                    axes[i, j].axvline(
                        dists_1d[j].median(), ls="--", color=color, lw=lw
                    )

    if label is not None:

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


def format_axes(axes, labels=None):
    """
    See the comments below for the individual adjustments made.
    
    axes:
        2D matplotlib.axes grid containing the marginal matrix
    """
    N = len(axes)

    # add labels
    labels = labels or [fr"$\theta_{i}$" for i in range(N)]
    assert N == len(labels)
    for i, label in enumerate(labels):
        axes[i, i].set_ylabel(f"$p$({labels[i]})")
        axes[len(labels) - 1, i].set_xlabel(label)

    # turn off top triangle
    for i in range(N):
        for j in range(i + 1, N):
            axes[i, j].axis("off")

    # adjust limits
    for i in range(N):
        for j in range(0, i + 1):
            xj = axes[j, j].lines[0].get_data()[0]
            axes[i, j].set_xlim(xj[0], xj[-1])
            if i != j:
                xi = axes[i, i].lines[0].get_data()[0]
                axes[i, j].set_ylim(xi[0], xi[-1])

    # remove all x tick labels but at the buttom row
    for i in range(N - 1):
        for j in range(0, i + 1):
            axes[i, j].xaxis.set_ticks([])

    # move all y tick labels to the very right plot of the row
    for i in range(N):
        axes[i, i].yaxis.tick_right()
        axes[i, i].yaxis.set_label_position("right")
        for j in range(0, i):
            axes[i, j].yaxis.set_ticks([])

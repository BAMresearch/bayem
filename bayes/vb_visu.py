import numpy as np

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
        x_min = np.min(dist.mean) * 0.02
        x_max = np.max(dist.mean) * 3
    # Create grid and multivariate normal
    x_plot = np.linspace(x_min, x_max, 5000)

    if expected_value is not None:
        expected_value = np.atleast_1d(expected_value)

    for i in range(len(dist.mean)):
        plt.plot(
            x_plot, dist.pdf(x_plot, i), label="%s of parameter nb %i" % (name1, i)
        )
        if expected_value is not None:
            plt.axvline(expected_value[i], ls=":", c="k")
        if "compare_with" in kwargs:
            plt.plot(
                x_plot,
                kwargs["compare_with"].pdf(x_plot, i),
                label="%s of parameter nb %i" % (name2, i),
            )

        plt.xlabel("Parameter")
        plt.legend()
        if plot == "individual":
            plt.show()
    if plot == "joint":
        plt.show()

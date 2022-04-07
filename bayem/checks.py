from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from .vba import DictModelError


def _measure(true, lin, norm):
    y_true, y_lin = np.asarray(true), np.asarray(lin)
    error = norm(y_true - y_lin)
    ref = norm(y_lin)
    if ref == 0:
        # Extremely unlikely case:
        # 1) model must not depend on parameters
        # 2) model must (depending on the norm used...) always be zero.
        #    IMO impossible, if there is any noise in the data.
        # Thus, we explicitly avoid the division and return nan and the user
        # should recognize an issue and investigate that case manually.
        return np.nan
    return error / ref


def linearity_analysis(
    model,
    posterior,
    sd_range=range(-3, 4),
    norm=np.linalg.norm,
    show=False,
    parameter_names=None,
    ret_values=False,
):
    """
    Compares the `model` responses `sd_range` standard deviations around the
    `posterior` mean with its linearization to estimate a measure of
    linearity.

    Use `ret_values=True` to obtain all raw values of the analyis for custom
    visualizations.
    """

    model = DictModelError(model, jac=None, noise0=None)

    k, J = model.first_call(posterior.mean)

    linearity_measure = defaultdict(dict)
    linearity_values = defaultdict(dict)

    if parameter_names is None:
        parameter_names = posterior.parameter_names

    std_diag = posterior.std_diag

    for name in parameter_names:
        i_sd = posterior.index(name)
        single_sd = np.zeros_like(posterior.mean)
        single_sd[i_sd] = std_diag[i_sd]

        prms = [posterior.mean + i * single_sd for i in sd_range]

        all_me_real = [model._Tk(model.f(prm)) for prm in prms]

        for noise in k.keys():

            me_lin = [k[noise] + J[noise] @ (prm - posterior.mean) for prm in prms]
            me_true = [me[noise] for me in all_me_real]
            linearity_values[noise][name] = me_true, me_lin
            linearity_measure[noise][name] = _measure(me_true, me_lin, norm=norm)

    if show:
        _debug_plot(linearity_values, sd_range, norm)

    # if the model error provides no noise group, we also remove noise groups
    # from the output
    linearity_measure = model.original_noise(linearity_measure)

    if ret_values:
        return linearity_measure, model.original_noise(linearity_values)
    else:
        return linearity_measure


def _debug_plot(linearity_values, sd_range, norm, ax=None):
    if ax is None:
        ax = plt.gca()
        show = True

    for noise in linearity_values:
        for name, (me_true, me_lin) in linearity_values[noise].items():
            y_true = [norm(m) for m in me_true]
            y_lin = [norm(m) for m in me_lin]

            line = ax.plot(sd_range, y_true, label=f"{name}, {noise}")[0]
            ax.plot(sd_range, y_lin, color=line.get_color(), ls="--")

    tick_labels = [f"µ{i:+4.2f}σ" for i in sd_range]
    ax.set_xticks(sd_range)
    ax.set_xticklabels(tick_labels)
    ax.legend()
    if show:
        plt.show()

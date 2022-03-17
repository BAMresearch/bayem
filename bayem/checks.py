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
):
    """
    Compares the `model` responses `sd_range` standard deviations around the
    `posterior` mean with its linearization to estimate a measure of
    linearity.
    """

    model = DictModelError(model, jac=None, noise0=None)

    k, J = model.first_call(posterior.mean)

    linearity_measure = defaultdict(dict)

    for i_sd, sd in enumerate(posterior.std_diag):
        name = posterior.parameter_names[i_sd]
        single_sd = np.zeros_like(posterior.mean)
        single_sd[i_sd] = sd

        prms = [posterior.mean + i * single_sd for i in sd_range]
        tick_labels = [f"µ{i:+4.2f}σ" for i in sd_range]

        for noise_group in k.keys():

            me_real = [model._Tk(model.f(prm))[noise_group] for prm in prms]

            me_lin = [
                k[noise_group] + J[noise_group] @ (prm - posterior.mean) for prm in prms
            ]

            if show:
                p = plt.plot(
                    sd_range,
                    [norm(model) for model in me_real],
                    label=f"{name}, noise group {noise_group}",
                )
                plt.plot(
                    sd_range,
                    [norm(model) for model in me_lin],
                    ls=":",
                    color=p[0].get_color(),
                )

            linearity_measure[noise_group][name] = _measure(me_real, me_lin, norm=norm)

    if show:
        plt.xticks(sd_range, tick_labels)
        plt.legend()
        plt.show()

    # if the model error provides no noise group, we also remove noise groups
    # from the output
    return model.original_noise(linearity_measure)

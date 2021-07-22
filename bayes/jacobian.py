import numpy as np


def delta_x(x0, delta=None):
    if delta is not None:
        return delta

    if x0 == 0:
        return 1.0e-6

    return x0 * 1.0e-6  # approx x0 * sqrt(machine precision)


def jacobian_cdf(f, latent_parameter_list, w_r_t_what=None):
    jac = dict()
    w_r_t_what = w_r_t_what or latent_parameter_list.names
    for prm_name in w_r_t_what:
        if hasattr(latent_parameter_list[prm_name], "__len__"):
            prm_jac = _vector_cdf(f, latent_parameter_list, prm_name)
        else:
            prm_jac = _scalar_cdf(f, latent_parameter_list, prm_name)

        for sensor_key, sensor_jac in prm_jac.items():
            if sensor_key not in jac:
                jac[sensor_key] = dict()
            jac[sensor_key][prm_name] = sensor_jac

    return jac


def _vector_cdf(f, latent_parameter_list, prm_name):
    jac = None
    value = latent_parameter_list[prm_name]
    N = len(value)

    for row in range(N):
        dx = delta_x(value[row])

        # generate N zeros with a single 1 at position row
        mask = np.eye(1, N, k=row).flatten()

        prm_plus = latent_parameter_list.with_value(prm_name, value + mask * 0.5 * dx)
        prm_minus = latent_parameter_list.with_value(prm_name, value - mask * 0.5 * dx)

        me_plus = f(prm_plus)
        me_minus = f(prm_minus)

        if jac is None:
            jac = dict()
            for sensor_key, me_values in me_plus.items():
                jac[sensor_key] = np.empty((len(me_values), N))

        for sensor_key in jac:
            jac[sensor_key][:, row] = (me_plus[sensor_key] - me_minus[sensor_key]) / dx

    return jac


def _scalar_cdf(f, latent_parameter_list, prm_name):
    jac = dict()
    value = latent_parameter_list[prm_name]

    dx = delta_x(value)
    prm_plus = latent_parameter_list.with_value(prm_name, value + 0.5 * dx)
    prm_minus = latent_parameter_list.with_value(prm_name, value - 0.5 * dx)

    me_plus = f(prm_plus)
    me_minus = f(prm_minus)

    for sensor_key in me_plus:
        jac[sensor_key] = (me_plus[sensor_key] - me_minus[sensor_key]) / dx

    return jac

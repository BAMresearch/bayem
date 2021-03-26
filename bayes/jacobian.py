import numpy as np


def delta_x(x0, delta=None):
    if delta is not None:
        return delta
    dx = x0 * 1.0e-7 + 1.0e-7  # approx x0 * sqrt(machine precision)
    if dx == 0:
        dx = 1.0e-7
    return dx


def d_model_error_d_named_parameter(model_error, prm_name):
    if hasattr(model_error.parameter_list[prm_name], "__len__"):
        return d_model_error_d_vector_parameter(model_error, prm_name)
    else:
        return d_model_error_d_scalar_parameter(model_error, prm_name)


def d_model_error_d_scalar_parameter(model_error, prm_name):
    """
    Calculates the derivative of `model_error` w.r.t the named parameter 
    `prm_name`.

    model_error:
        object that has an attribute `parameter_list` and a __call__() method
        without arguments that returns a dict of type 
        {key : numpy_vector of length N}
    prm_name:
        name of a named scalar parameter in `model_error.parameter_list` 
    returns:
        dict of type {key : numpy_vector of length N}
    """
    prm0 = model_error.parameter_list[prm_name]
    dx = delta_x(prm0)

    model_error.parameter_list[prm_name] = prm0 - dx
    me0 = model_error()
    model_error.parameter_list[prm_name] = prm0 + dx
    me1 = model_error()
    model_error.parameter_list[prm_name] = prm0

    jac = dict()
    for key in me0:
        jac[key] = (me1[key] - me0[key]) / (2 * dx)
    return jac


def d_model_error_d_vector_parameter(model_error, prm_name):
    """
    Calculates the derivative of `model_error` w.r.t the named parameter 
    `prm_name`.

    model_error:
        object that has an attribute `parameter_list` and a __call__() method
        without arguments that returns a dict of type 
        {key : numpy_vector of length N}
    prm_name:
        name of a named vector parameter in `model_error.parameter_list` of 
        length M
    returns:
        dict of type {key : numpy_matrix of length NxM}
    """
    prm0 = np.copy(model_error.parameter_list[prm_name])
    M = len(prm0)
    jac = dict()

    for row in range(M):
        dx = delta_x(prm0[row])

        model_error.parameter_list[prm_name][row] = prm0[row] - dx
        me0 = model_error()
        model_error.parameter_list[prm_name][row] = prm0[row] + dx
        me1 = model_error()
        model_error.parameter_list[prm_name][row] = prm0[row]

        for key in me0:
            if key not in jac:
                jac[key] = np.empty((len(me0[key]), M))

            jac[key][:, row] = (me1[key] - me0[key]) / (2 * dx)
    return jac

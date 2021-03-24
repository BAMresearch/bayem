import numpy as np


def d_model_error_d_vector(model_error, number_vector):
    """
    Calculates the derivative of `model_error` w.r.t `number_vector`

    model_error:
        function that takes the single argument of type `number_vector` and
        returns a dict of type {key : numpy_vector of length N}
    number_vector:
        vector of numbers of length M
    returns:
        dict of type {key : numpy_matrix of shape NxM}
    """
    x = np.copy(number_vector)

    for iParam in range(len(x)):
        dx = x[iParam] * 1.0e-7  # approx x0 * sqrt(machine precision)
        if dx == 0:
            dx = 1.0e-10

        x[iParam] -= dx
        fs0 = model_error(x)
        x[iParam] += 2 * dx
        fs1 = model_error(x)
        x[iParam] -= dx

        if iParam == 0:
            # allocate jac
            jac = {}
            for noise_key, f0 in fs0.items():
                jac[noise_key] = np.empty([len(f0), len(x)])

        for n in fs0:
            jac[n][:, iParam] = -(fs1[n] - fs0[n]) / (2 * dx)

    return jac


def d_model_error_d_scalar_parameter(model_error, prm_name):
    """
    Calculates the derivative of `model_error` w.r.t the named parameter 
    `prm_name`.

    model_error:
        object that has an attribute `parameter_list` and a __call__() method
        that returns a dict of type {key : numpy_vector of length N}
    prm_name:
        name of a named scalar parameter in `model_error.parameter_list` 
    returns:
        dict of type {key : numpy_vector of length N}
    """
    prm0 = model_error.parameter_list[prm_name]
    dx = prm0 * 1.0e-7  # approx prm * sqrt(machine precision)
    if dx == 0:
        dx = 1.0e-7

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
        that returns a dict of type {key : numpy_vector of length N}
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
        dx = prm0[row] * 1.0e-7  # approx prm * sqrt(machine precision)
        if dx == 0:
            dx = 1.0e-7

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

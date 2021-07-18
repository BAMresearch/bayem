# standard library imports
import copy as cp

# third party imports
import numpy as np

# local imports
from jacobian import delta_x
from subroutines import len_or_one


class ModelTemplate:

    def __init__(self, prms_def):
        """
        Parameters
        ----------
        prms_def : list
            A list of strings defining how a model parameter vector given to
            the __call__ method is interpreted. E.g. prms_def = ['a', 'b']
            means that the model parameter vector has two elements, the first of
            which gives the value of 'a' and the second gives the value of 'b'.
        """
        self.prms_def = prms_def
        self.theta_dim = len_or_one(prms_def)

    def __call__(self, x, theta):
        """
        Evaluates the model at x and theta. This method has to be overwritten
        by the user's model class.

        Parameters
        ----------
        x : array_like
            The input data of the model, i.e. the experimental input data.
        theta : array_like
            The parameter vector of the model.

        Returns
        -------
        jac : array_like
            The gradient of the model function at x, theta wrt theta
        """
        raise NotImplementedError(
            "Your model does not have a __call__-method. You need to define" +
            "this method so you can evaluate your model."
        )

    def jacobian(self, x, theta):
        """
        Computes the gradient of the model function at x and theta wrt theta.

        Parameters
        ----------
        x : array_like
            The input data of the model, i.e. the experimental input data.
        theta : array_like
            The parameter vector of the model.

        Returns
        -------
        jac : array_like
            The gradient of the model function at x, theta wrt theta
        """
        # allocate the output array
        jac = np.zeros(self.theta_dim)
        for i, theta_i in enumerate(theta):
            # evaluate the model at theta_i - h
            h = delta_x(theta_i)
            theta_left = cp.copy(theta)
            theta_left[i] = theta_i - h
            left = self(x, theta_left)
            # evaluate the model at theta_i + h
            theta_right = cp.copy(theta)
            theta_right[i] = theta_i + h
            right = self(x, theta_right)
            # evaluate the symmetric difference scheme
            jac[i] = (right - left) / (2 * h)
        return jac

    def error_function(self, x, theta, ye):
        ym = self(x, theta)
        error = ym - ye
        return error

    def error(self, theta, experiments, key="sensor"):

        model_error = {}
        if key == "experiment":
            for experiment_name, experiment_dict in experiments.items():
                x = experiment_dict['input']['value']
                ye = experiment_dict['output']['value']
                model_error[experiment_name] = self.error_function(x, theta, ye)
        elif key == "sensor":
            for experiment_name, experiment_dict in experiments.items():
                output_sensor = experiment_dict['output']['sensor']
                x = experiment_dict['input']['value']
                ye = experiment_dict['output']['value']
                me = self.error_function(x, theta, ye)
                if output_sensor not in model_error.keys():
                    model_error[output_sensor] = [me]
                else:
                    model_error[output_sensor].append(me)

        return model_error


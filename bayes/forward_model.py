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
        self.prms_dim = len_or_one(prms_def)

    def __call__(self, x, prms):
        """
        Evaluates the model at x and prms. This method has to be overwritten
        by the user's model class.

        Parameters
        ----------
        x : array_like
            The input data of the model, i.e. the experimental input data.
        prms : array_like
            The parameter vector of the model.

        Returns
        -------
        jac : array_like
            The gradient of the model function at x, prms wrt prms
        """
        raise NotImplementedError(
            "Your model does not have a __call__-method. You need to define" +
            "this method so you can evaluate your model."
        )

    def jacobian(self, x, prms):
        """
        Computes the gradient of the model function at x and prms wrt prms.

        Parameters
        ----------
        x : array_like
            The input data of the model, i.e. the experimental input data.
        prms : array_like
            The parameter vector of the model.

        Returns
        -------
        jac : array_like
            The gradient of the model function at x, prms wrt prms
        """
        # allocate the output array
        jac = np.zeros(self.prms_dim)
        for i, prms_i in enumerate(prms):
            # evaluate the model at prms_i - h
            h = delta_x(prms_i)
            prms_left = cp.copy(prms)
            prms_left[i] = prms_i - h
            left = self(x, prms_left)
            # evaluate the model at prms_i + h
            prms_right = cp.copy(prms)
            prms_right[i] = prms_i + h
            right = self(x, prms_right)
            # evaluate the symmetric difference scheme
            jac[i] = (right - left) / (2 * h)
        return jac

    def error_function(self, x, prms, ye):
        """
        Evaluates the model error for a single experiment. This function can be
        overwritten if another definition of the model error should be appllied.

        Parameters
        ----------
        x : array_like
            The input data of the model, i.e. the experimental input data.
        prms : array_like
            The parameter vector of the model.
        ye : array_like
            The measured output of the considered experiment.
        """
        ym = self(x, prms)
        error = ym - ye
        return error

    def error(self, prms, experiments, key="sensor"):
        """
        Computes the model error for all given experiments and returns them in
        a dictionary that is sorted with respect to the given key.

        Parameters
        ----------
        prms : array_like
            The parameter vector of the model.
        experiments : dict
            A dictionary with a structure like self._experiments in the class
            InferenceProblem.
        key : string, optional
            Either 'sensor' or 'experiment' defining how the model_error
            dictionary should be sorted.

        Returns
        -------
        model_error : dict
            A dictionary keys defined by the key argument of this method, and
            lists of numbers representing the model errors as values.
        """

        # check the input
        if key not in ["sensor", "experiment"]:
            raise RuntimeError(
                f"This method requires key='sensor' or key='experiment' as " +
                f"argument. You defined key='{key}'"
            )

        # fill the model_error dictionary in a structure defined by the given
        # 'key'-argument (either by experiment or sensor)
        model_error = {}
        if key == "experiment":
            # the keys of model_error will be the experiment names
            for experiment_name, experiment_dict in experiments.items():
                x = experiment_dict['input']['value']
                ye = experiment_dict['output']['value']
                model_error[experiment_name] = self.error_function(x, prms, ye)
        elif key == "sensor":
            # the keys of model_error will be the sensor types
            for experiment_name, experiment_dict in experiments.items():
                output_sensor = experiment_dict['output']['sensor']
                x = experiment_dict['input']['value']
                ye = experiment_dict['output']['value']
                me = self.error_function(x, prms, ye)
                if output_sensor not in model_error.keys():
                    model_error[output_sensor] = [me]
                else:
                    model_error[output_sensor].append(me)

        return model_error


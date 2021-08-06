# standard library imports
import copy as cp

# third party imports
import numpy as np

# local imports
from bayes.jacobian import delta_x
from bayes.subroutines import len_or_one


class OutputSensor:

    def __init__(self, name):
        """
        Parameters
        ----------
        name : string
            The name of the sensor, e.g. 'Deflection-Sensor bottom-left'.
        """
        self.name = name


class ModelTemplate:

    def __init__(self, prms_def, output_sensors):
        """
        Parameters
        ----------
        prms_def : list
            A list of strings defining how a model parameter vector given to
            the __call__ method is interpreted. E.g. prms_def = ['a', 'b']
            means that the model parameter vector has two elements, the first of
            which gives the value of 'a' and the second gives the value of 'b'.
        output_sensors : list
            Contains sensor-objects which must at least a 'name' attribute.
        """
        self.prms_def = prms_def
        self.prms_dim = len_or_one(prms_def)
        self.output_sensors = output_sensors

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

    def error_function(self, x, prms, ye, output_sensor):
        """
        Evaluates the model error for a single experiment. This function can be
        overwritten if another definition of the model error should be applied.

        Parameters
        ----------
        x : array_like
            The input data of the model, i.e. the experimental input data.
        prms : ParameterList-object
            Dictionary-like object containing parameter name:value pairs.
        ye : array_like
            The measured output of the considered experiment.
        output_sensor : string
            The name of the sensor, the error should be evaluated for.
        """
        ym = self(x, prms)[output_sensor]
        error = ym - ye
        return error

    def error(self, prms, experiments):
        """
        Computes the model error for all given experiments and returns them in
        a dictionary that is sorted by output sensor.

        Parameters
        ----------
        prms : ParameterList-object
            Dictionary-like object containing parameter name:value pairs
        experiments : dict
            A dictionary with a structure like self._experiments in the class
            InferenceProblem.

        Returns
        -------
        model_error : dict
            A dictionary with the keys being the output sensor names, and lists
            lists of numbers representing the model errors as values.
        """
        model_error = {}
        for exp_dict in experiments.values():
            inp = exp_dict['input']
            output_sensors = exp_dict['output']
            for output_sensor, ye in output_sensors.items():
                me = self.error_function(inp, prms, ye, output_sensor)
                if output_sensor not in model_error.keys():
                    model_error[output_sensor] = [me]
                else:
                    model_error[output_sensor].append(me)
        return model_error

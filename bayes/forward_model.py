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

    def __call__(self, inp, prms):
        """
        Evaluates the model response for each output sensor and returns the
        response dictionary.

        Parameters
        ----------
        inp : array_like
            The input data of the model, i.e. the experimental input data.
        prms : array_like
            The parameter vector of the model.

        Returns
        -------
        response_dict : dict
            Contains the model response (value) for each output sensor,
            referenced by their name (key).
        """
        response_dict = {}
        for output_sensor in self.output_sensors:
            response_dict[output_sensor.name] =\
                self.response({**inp, **prms}, output_sensor)
        return response_dict

    def response(self, inp, sensor):
        """
        Evaluates the model at x and prms for the given sensor. This method has
        to be overwritten by the user's response method.

        Parameters
        ----------
        inp : ParameterList object
            Both the input data and the parameters of the model.
        sensor : object
            The output sensor the response should be evaluated for.
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

    @staticmethod
    def apply_sensor_error(sensors, prms, se_dict):
        """
        Accounts for additive or multiplicative sensor error by removing the
        respective error.

        Parameters
        ----------
        sensors : dict
            Contains the experimental input or output sensors with values
        prms : ParameterList-object
            Dictionary-like object containing parameter name:value pairs.
        se_dict : dict
            Contains information on sensors for which errors have been defined

        Returns
        -------
        sen_copy : dict
            A copy of sensors with adjusted values for the respective sensors.
        """
        sen_copy = cp.copy(sensors)
        for sensor_name in sen_copy.keys():
            if sensor_name in se_dict.keys():
                if se_dict[sensor_name]['rel']:
                    factor = 1 + prms[se_dict[sensor_name]['prm']]
                    sen_copy[sensor_name] /= factor
                else:
                    summand = prms[se_dict[sensor_name]['prm']]
                    sen_copy[sensor_name] -= summand
        return sen_copy

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

    def error(self, prms, experiments, se_dict):
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
        se_dict : dict
            Contains information on sensors for which errors have been defined

        Returns
        -------
        model_error : dict
            A dictionary with the keys being the output sensor names, and lists
            lists of numbers representing the model errors as values.
        """
        model_error = {}
        for exp_dict in experiments.values():
            if not se_dict:
                inp = exp_dict['input']
                output_sensors = exp_dict['output']
            else:
                inp = self.apply_sensor_error(exp_dict['input'], prms, se_dict)
                output_sensors = self.apply_sensor_error(exp_dict['output'],
                                                         prms, se_dict)
            for output_sensor, ye in output_sensors.items():
                me = self.error_function(inp, prms, ye, output_sensor)
                if output_sensor not in model_error.keys():
                    model_error[output_sensor] = [me]
                else:
                    model_error[output_sensor].append(me)
        return model_error

# standard library imports
import copy as cp

# local imports
from bayes.jacobian import delta_x
from bayes.subroutines import len_or_one
from bayes.subroutines import list2dict


class OutputSensor:
    """
    Base class for an output sensor of the forward model. Each model response
    of the forward model needs to be associated with an output sensor. At least
    such an object must have a name and an error_metric, but can also have
    additional attributes such as its position or its temperature. In these
    cases the user has to define his own output sensor class, which should be
    derived from this one.
    """
    def __init__(self, name, error_metric='abs'):
        """
        Parameters
        ----------
        name : string
            The name of the sensor, e.g. 'Deflection-Sensor bottom-left'.
        error_metric : string, optional
            Either 'abs' (absolute) or 'rel' (relative). Defines if the model
            error with respect to this sensor should be measured in absolute or
            relative terms. In the former case ('abs'), the error is defined as
            (model prediction - measured value). In the latter case ('rel') the
            error is defined as (1 - (model prediction / measured value)).
        """
        self.name = name
        self.error_metric = error_metric


class ModelTemplate:
    """
    This class serves as a base class for any forward model. When you want to
    define a specific forward model, you need to derive your own class from this
    one, and then define the 'response' method. The latter essentially describes
    the model function mapping the model input to the output.
    """
    def __init__(self, prms_def_, output_sensors):
        """
        Parameters
        ----------
        prms_def_ : list
            Contains the model's calibration parameter names. The list may only
            contain strings or one-element dictionaries. It could look, for
            example, like [{'a': 'm'}, 'b']. The one-element dictionaries
            account for the possibility to define a local name for a calibration
            parameter that is different from the global name. In the example
            above, the calibration parameter with the global name 'a' will be
            referred to as 'm' within the model. So, the one-element dicts have
            the meaning {<global name>: <local name>}. String-elements are
            interpreted as having similar local and global names. Note that the
            local-name option will not be required most of the times.
        output_sensors : list[OutputSensor]
            Contains sensor-objects structuring the model output.
        """

        # convert the given parameter names to a dictionary with global names
        # as keys and local names as values
        self.prms_def = list2dict(prms_def_)

        # other attributes
        self.prms_dim = len_or_one(self.prms_def)
        self.output_sensors = output_sensors

        # this dictionary allows to look up an output sensor's error metric by
        # the output sensor's name
        self.error_metric_dict = {}
        for output_sensor in output_sensors:
            self.error_metric_dict[output_sensor.name] =\
                output_sensor.error_metric

    def __call__(self, exp_inp, prms):
        """
        Evaluates the model response for each output sensor and returns the
        response dictionary.

        Parameters
        ----------
        exp_inp : dict
            The experimental input data of the model. The keys are names of the
            experiment's input sensors, and the values are their numeric values.
        prms : dict
            The calibration parameters with their local names as keys and their
            numeric values as values.

        Returns
        -------
        response_dict : dict
            Contains the model response (value) for each output sensor,
            referenced by the output sensor's name (key).
        """
        response_dict = {}
        for output_sensor in self.output_sensors:
            # note that {**d1, **d2} adds the two dictionaries d1 and d2
            response_dict[output_sensor.name] =\
                self.response({**exp_inp, **prms}, output_sensor)
        return response_dict

    def response(self, inp, sensor):
        """
        Evaluates the model for the given sensor. This method has to be
        overwritten by the user's response method.

        Parameters
        ----------
        inp : dict
            Contains both the exp. input data and the  model's parameters. The
            keys are the names, and the values are their numeric values.
        sensor : object
            The output sensor the response should be evaluated for.
        """
        raise NotImplementedError(
            "Your model does not have a proper response-method yet. You need " +
            "to define this method, so you can evaluate your model."
        )

    def jacobian(self, exp_inp, prms):
        """
        Computes the gradient of the response function at exp_inp and prms with
        respect to the calibration parameters prms. A symmetric difference
        scheme is used here.

        Parameters
        ----------
        exp_inp : dict
            The experimental input data of the model. The keys are names of the
            experiment's input sensors, and the values are their numeric values.
        prms : dict
            The calibration parameters with their local names as keys and their
            numeric values as values.

        Returns
        -------
        jac : dict
            Contains the calibration parameter names as keys. The values are
            dictionaries with the model's output sensor names as keys and the
            respective approximated derivatives as values.
        """
        jac = {}
        for prm_name, prm_value in prms.items():
            # evaluate the model at prms_i - h
            h = delta_x(prm_value)
            prms_left = cp.copy(prms)
            prms_left[prm_name] = prm_value - h
            left = self(exp_inp, prms_left)
            # evaluate the model at prms_i + h
            prms_right = cp.copy(prms)
            prms_right[prm_name] = prm_value + h
            right = self(exp_inp, prms_right)
            # evaluate the symmetric difference scheme
            jac[prm_name] = {}
            for sensor_name in left.keys():
                jac[prm_name][sensor_name] =\
                    (right[sensor_name] - left[sensor_name]) / (2 * h)
        return jac

    @staticmethod
    def apply_sensor_error(sensors, prms, sensor_error_dict):
        """
        Accounts for additive or multiplicative sensor error (understood here as
        a sensor bias). Note that these sensor errors are understood as unknown,
        and therefore modelled as calibration parameters to be inferred.

        Parameters
        ----------
        sensors : dict
            Contains the experimental input or output sensors with their names
            as keys and their numeric values as values.
        prms : dict
            The calibration parameters with their local names as keys and their
            numeric values as values.
        sensor_error_dict : dict
            Contains information on sensors for which errors have been defined.
            A key-value pair in this dictionary has the following structure:
            <sensor name>: {'rel': <bool>, 'prm': <string>}. Here, the <sensor
            name> defines the sensor an error has been defined for. The 'rel'
            item defines if the error is relative ('rel' is True) or absolute
            ('rel' is False). Finally, the 'prm' item defines the calibration
            parameter the bias is described with.

        Returns
        -------
        sensor_copy : dict
            A copy of the provided input argument 'sensors' with adjusted values
            for the sensors that are associated with sensor errors.
        """
        # we don't want side effects to modify sensors outside of this method;
        # this is why we make a copy here
        sensor_copy = cp.copy(sensors)
        for sensor_name in sensor_copy.keys():
            # the sensor_error_dict.keys() in the line below contain the sensor
            # names for which errors have been defined; can also be empty
            if sensor_name in sensor_error_dict.keys():
                # now apply the correct sensor error (relative or absolute)
                if sensor_error_dict[sensor_name]['rel']:
                    factor = 1 + prms[sensor_error_dict[sensor_name]['prm']]
                    # the idea here is to reverse the relative, meaning
                    # multiplicative, error; since the error is applied by
                    # multiplication, its reversal needs the respective division
                    sensor_copy[sensor_name] /= factor
                else:
                    summand = prms[sensor_error_dict[sensor_name]['prm']]
                    # the idea here is to reverse the absolute, meaning additive
                    # error; since the error is applied by addition, its
                    # reversal needs the respective subtraction
                    sensor_copy[sensor_name] -= summand
        return sensor_copy

    def error_function(self, exp_inp, prms, output_sensor_name, ye):
        """
        Evaluates the model error for a single experiment. This function can be
        overwritten if another definition of the model error should be applied.

        Parameters
        ----------
        exp_inp : dict
            The experimental input data of the model. The keys are names of the
            experiment's input sensors, and the values are their numeric values.
        prms : dict
            The calibration parameters with their local names as keys and their
            numeric values as values.
        output_sensor_name : string
            The name of the model's output sensor, the error should be evaluated
            for.
        ye : float
            The experimental output value for the given output_sensor

        Returns
        -------
        error : float
            The computed model error for the given output sensor.
        """

        # compute the model prediction for the given sensor
        ym = self(exp_inp, prms)[output_sensor_name]

        # compute the error according to the defined error metric
        if self.error_metric_dict[output_sensor_name] == 'abs':
            error = ym - ye
        elif self.error_metric_dict[output_sensor_name] == 'rel':
            error = 1.0 - ym / ye
        else:
            raise ValueError(
                f"Output sensor '{output_sensor_name}' has an unknown error "
                f"metric: '{self.error_metric_dict[output_sensor_name]}'."
            )

        return error

    def error(self, prms, experiments, sensor_error_dict):
        """
        Computes the model error for all given experiments and returns them in
        a dictionary that is sorted by output sensors.

        Parameters
        ----------
        prms : dict
            The calibration parameters with their local names as keys and their
            numeric values as values.
        experiments : dict
            Contains the experiment names (strings) as keys, while the values
            are dicts with the structure {'input': <dict>, 'output': <dict>,
            'forward_model': <string>}. The 'input' value is a dictionary of
            input sensors (<sensor name>: <sensor value> pairs), while the
            output value is a similar dictionary of output sensors. The
            remaining 'forward_model' value states the forward model's name, but
            is not used here.
        sensor_error_dict : dict
            Contains information on sensors for which errors have been defined.
            A key-value pair in this dictionary has the following structure:
            <sensor name>: {'rel': <bool>, 'prm': <string>}. Here, the <sensor
            name> defines the sensor an error has been defined for. The 'rel'
            item defines if the error is relative ('rel' is True) or absolute
            ('rel' is False). Finally, the 'prm' item defines the calibration
            parameter the bias is described with.

        Returns
        -------
        model_error : dict
            A dictionary with the keys being the output sensor names, and lists
            of numbers representing the model errors as values.
        """
        model_error_dict = {}
        # loop over all experiments defined within the problem; note that the
        # 'if' condition in the next line is placed outside the loop
        # intentionally, so that it is not evaluated each run in the loop
        if not sensor_error_dict:
            # in this case, no sensor errors are defined
            for exp_dict in experiments.values():
                exp_inp = exp_dict['input']
                exp_out = exp_dict['output']
                for output_sensor_name, ye in exp_out.items():
                    me = self.error_function(exp_inp, prms,
                                             output_sensor_name, ye)
                    if output_sensor_name not in model_error_dict.keys():
                        model_error_dict[output_sensor_name] = [me]
                    else:
                        model_error_dict[output_sensor_name].append(me)
        else:
            # in this case, there is at least one sensor with a defined error
            for exp_dict in experiments.values():
                exp_inp = self.apply_sensor_error(exp_dict['input'], prms,
                                                  sensor_error_dict)
                exp_out = self.apply_sensor_error(exp_dict['output'], prms,
                                                  sensor_error_dict)
                for output_sensor_name, ye in exp_out.items():
                    me = self.error_function(exp_inp, prms,
                                             output_sensor_name, ye)
                    if output_sensor_name not in model_error_dict.keys():
                        model_error_dict[output_sensor_name] = [me]
                    else:
                        model_error_dict[output_sensor_name].append(me)

        return model_error_dict

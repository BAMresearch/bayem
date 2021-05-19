import numpy as np
from .parameters import ParameterList

"""
The inference problem provides:
    model_error_dict:
        {model_error_key: {sensor: vector of model_error}}
    jacobian_dict:
        {model_error_key: {sensor: {parameters : vector/matrix of jacobian}}}

Both the loglikelihood function and VB require an input that is sorted by a
`noise group`. The conversion (basically a concatenation) of various pairs of
(model_error_key, sensor) is done via the `NoiseModelInterface`.

Example:
========
    See test/test_noise.py for an exact implementation of the following example.

    `TrussModelError` with model_error_key `truss_key` returns
        {ForceSensor: [1,2,3], TemperatureSensor: [20]} 

    and `BeamModelError` with model_error_key `beam_key` returns
        {ForceSensor: [10, 20, 30], InclinationSensor: [0.42]}

    Thus, the model_error_dict of a inference problem would look like:
        {
         truss_key:
            {ForceSensor: [1,2,3], TemperatureSensor: [20]},
         beam_key:
            {ForceSensor: [10, 20, 30], InclinationSensor: [0.42]}
        }


    We now want three different noise terms. One for the two `ForceSensor`s, 
    one for `TemperatureSensor` and one for `InclinationSensor`.

    So we basically assign:
        
        noise_force : [(truss_key, ForceSensor), (beam_key, ForceSensor)]
        noise_temp  : [(truss_key, TemperatureSensor)]
        noise_incl  : [(beam_key, InclinationSensor)]

    such that the model_error_dict is rearranged to:

        {
         noise_force : [1,2,3,10,20,30],
         noise_temp  : [20],
         noise_incl  : [0.42]
        }

    For convenience, there are multiple NoiseModels defined to simplify this
    definition.

    Also, there is a `split` method that converts an input like
        noise_force : [1,2,3,10,20,30]
    back to its individual contributions
        {
         truss_key: {ForceSensor: [1,2,3]}
         beam_key: {ForceSensor: [10, 20, 30]}
        }

"""


class NoiseModelInterface:
    def vector_contribution(self, model_error_dict):
        """
        Rearranges the (model_error_key, sensor) ordering of 
        `model_error_dict` a single numpy vector. 
        """
        raise NotImplementedError("Implement me!")

    def jacobian_contribution(self, jacobian_dict):
        """
        Rearranges the (model_error_key, sensor) ordering of 
        `jacobian_dict` a single numpy matrix. 
        """
        raise NotImplementedError("Implement me!")

    def loglike_contribution(self, model_error_dict):
        """
        Rearranges the (model_error_key, sensor) ordering of 
        `model_error_dict` a single numpy matrix and calculates its
        contribution to a loglikelihood function.
        """
        raise NotImplementedError("Implement me!")


class UncorrelatedNoiseModel(NoiseModelInterface):
    def __init__(self):
        self.parameter_list = ParameterList()
        self.parameter_list.define("precision")
        self._terms = []
        self._lengths = []  # of the terms for a potential `self.split`

    def add(self, sensor, model_error_key=None):
        """
        Adds a (`sensor`, `model_error_key`) pair to the noise model such that 
        the corresponding output, e.g.
            model_error_dict[model_error_key][sensor]
        or 
            jacobian_dict[model_error_key][sensor]
        is added to the `self.vector_contribution` or 
        `self.jacobian_contribution`, respectively.
        """
        self._terms.append((sensor, model_error_key))

    def _define_terms(self, model_error_dict):
        """
        Can be overwritten to automatically define `self._terms`
        for certain special cases. See `UncorrelatedSensorNoise` or
        `UncorrelatedSingleNoise` below.
        """
        pass

    def vector_contribution(self, model_error_dict):
        """
        overwritten
        """
        return self._stack(model_error_dict)

    def jacobian_contribution(self, jacobian_dict):
        """
        overwritten
        """
        return self._stack(jacobian_dict)

    def loglike_contribution(self, model_error_dict):
        """
        overwritten
        """
        error = self.vector_contribution(model_error_dict)
        sigma = 1.0 / self.parameter_list["precision"] ** 0.5
        return -0.5 * (
            len(error) * np.log(2.0 * np.pi * sigma ** 2)
            + np.sum(np.square(error / sigma ** 2))
        )

    def _stack(self, dict_of_dicts):
        """
        Extracts the (sensor, model_error_key) from the nested `dict_of_dicts`
        (could be both `model_error_dict` or `jacobian_dict`) and concatenates
        them to a long numpy vector/matrix.

        It keeps track of the dimension (self._lengths) of the individual terms
        to reverse this operation in `self.split()`.
        """
        self._define_terms(dict_of_dicts)
        terms = []

        self._lengths = []
        for (sensor, key) in self._terms:
            term = dict_of_dicts[key][sensor]
            self._lengths.append(len(term))
            terms.append(term)
        return np.concatenate(terms)

    def split(self, array):
        """
        Reverse operation of `self._stack`. 

        Use case:
            You propagate the uncertainty via
                J = jacobian_contribution(mean)
                variance = J @ inferred_cov @ J.T
                sd = sqrt(diag(variance))
            and now want to split up the purely numeric sd back to the individual
            terms.
        """
        if not self._lengths:
            raise RuntimeError(
                "You have to evaluate (e.g. call .vector_contribution) at least once before attemting a .split()!"
            )

        total_length = sum(self._lengths)
        if total_length != len(array):
            raise RuntimeError(
                f"Dimension mismatch: The noise model is of length {total_length}, you provided length {len(array)}."
            )

        splitted = {}
        offset = 0
        for length, (sensor, key) in zip(self._lengths, self._terms):
            if key not in splitted:
                splitted[key] = {}

            term = array[offset : offset + length]
            splitted[key][sensor] = term

            offset += length

        return splitted


class UncorrelatedSingleNoise(UncorrelatedNoiseModel):
    """
    Noise model with single term for _all_ contributions of the model error.
    """

    def _define_terms(self, model_error_dict):
        if self._terms:
            return

        for me_key, me in model_error_dict.items():
            for sensor in me:
                self.add(sensor, me_key)


class UncorrelatedSensorNoise(UncorrelatedNoiseModel):
    """
    Uncorrelated noise term that allows to specify exactly which sensors
    from the model error are taken for the term.
    """

    def __init__(self, sensors):
        super().__init__()
        if type(sensors) is list or type(sensors) is tuple:
            self.sensors = sensors
        else:
            self.sensors = [sensors]

    def _define_terms(self, model_error_dict):
        if self._terms:
            return

        for me_key, me in model_error_dict.items():
            for sensor in me:
                if sensor in self.sensors:
                    self.add(sensor, me_key)

import numpy as np
from .parameters import ParameterList
import math

"""
The inference problem provides:
    model_error_dict:
        {model_error_key: {sensor: vector of model_error}}
    jacobian_dict:
        {model_error_key: {sensor: {parameters : vector/matrix of jacobian}}}

Both the loglikelihood function and VB require an input that is sorted by a
`noise group`. The conversion (basically a concatenation) of various pairs of
(model_error_key, sensor) is done via the `NoiseModelInterface`.

See test/test_noise.py for an example and further explaination.
"""


class NoiseModelInterface:
    def model_error_terms(self, model_error_dict):
        """
        Rearranges the (model_error_key, sensor) ordering of 
        `model_error_dict` into a single numpy vector. 
        """
        raise NotImplementedError("Implement me!")

    def jacobian_terms(self, jacobian_dict):
        """
        Rearranges the (model_error_key, sensor) ordering of 
        `jacobian_dict` into a single numpy matrix. 
        """
        raise NotImplementedError("Implement me!")

    def loglike_contribution(self, model_error_dict):
        """
        Rearranges the (model_error_key, sensor) ordering of 
        `model_error_dict` into a single numpy vector and calculates its
        contribution to a loglikelihood function.
        """
        raise NotImplementedError("Implement me!")


class UncorrelatedNoiseModel(NoiseModelInterface):
    def __init__(self):
        self.parameter_list = ParameterList()
        self.parameter_list.define("precision")
        self._key_pairs = []

    def add(self, model_error_key, sensor):
        """
        Adds a (`model_error_key`, `sensor`) pair to the noise model such that 
        the corresponding output, e.g.
            model_error_dict[model_error_key][sensor]
        or 
            jacobian_dict[model_error_key][sensor]
        is added to the `self.model_error_terms` or 
        `self.jacobian_terms`, respectively.
        """
        self._key_pairs.append((model_error_key, sensor))

    def _define_key_pairs(self, model_error_dict):
        """
        Can be overwritten to automatically define `self._key_pairs`
        for certain special cases. See `UncorrelatedSensorNoise` or
        `UncorrelatedSingleNoise` below.
        """
        pass

    def model_error_terms(self, model_error_dict):
        """
        overwritten
        """
        return self._by_noise(model_error_dict)

    def jacobian_terms(self, jacobian_dict):
        """
        overwritten
        """
        return self._by_noise(jacobian_dict)

    def loglike_contribution(self, model_error_dict, noise_prm):
        """
        overwritten
        """
        prec = self.parameter_list.overwrite_with(noise_prm)["precision"]
        terms = self.model_error_terms(model_error_dict)
        ll = 0.0
        for error in terms:
            ll -= len(error)/2 * math.log(2*math.pi/prec)
            ll -= 0.5 * prec * np.sum(np.square(error))
        return ll

    def _by_noise(self, dict_of_dicts):
        """
        Extracts the (sensor, model_error_key) from the nested `dict_of_dicts`
        (could be both `model_error_dict` or `jacobian_dict`) and concatenates
        them to a long numpy vector/matrix.

        """
        self._define_key_pairs(dict_of_dicts)
        terms = [dict_of_dicts[key][sensor] for (key, sensor) in self._key_pairs]
        return terms

    def by_keys(self, terms):
        """
        Reverse operation of `self._by_noise`. 
        """

        if len(self._key_pairs) != len(terms):
            raise RuntimeError(
                f"Dimension mismatch: The noise model has {len(self._key_pairs)} terms, you provided {len(terms)}."
            )

        splitted = {}
        for term, (key, sensor) in zip(terms, self._key_pairs):
            if key not in splitted:
                splitted[key] = {}

            splitted[key][sensor] = term

        return splitted


class UncorrelatedSingleNoise(UncorrelatedNoiseModel):
    """
    Noise model with single term for _all_ contributions of the model error.
    """

    def _define_key_pairs(self, model_error_dict):
        if self._key_pairs:
            return

        for me_key, me in model_error_dict.items():
            for sensor in me:
                self.add(me_key, sensor)


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

    def _define_key_pairs(self, model_error_dict):
        if self._key_pairs:
            return

        for me_key, me in model_error_dict.items():
            for sensor in me:
                if sensor in self.sensors:
                    self.add(me_key, sensor)

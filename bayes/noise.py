import numpy as np
from .parameters import ParameterList


class NoiseModelInterface:
    def __init__(self):
        self.parameter_list = ParameterList()

    def vector_contribution(self, raw_me):
        raise NotImplementedError()

    def loglike_contribution(self, raw_me):
        raise NotImplementedError()

    def _loglike_term(self, error, sigma):
        return -0.5 * (
            len(error) * np.log(2.0 * np.pi * sigma ** 2)
            + np.sum(np.square(error / sigma ** 2))
        )


class SingleSensorNoise(NoiseModelInterface):
    """
    Noise model with single term for _all_ contributions of the model error.
    """

    def vector_contribution(self, raw_me):
        vector_terms = []
        for exp_me in raw_me.values():
            if not isinstance(exp_me, dict):
                raise RuntimeError(
                    "The `SingleSensorNoise` model assumes that your model "
                    "error returns a dict {some_key : numbers}, but yours did "
                    "not. Use `SingleNoise` instead."
                )
            for sensor_me in exp_me.values():
                vector_terms.append(sensor_me)
        return np.concatenate(vector_terms)


class SingleNoise(NoiseModelInterface):
    """
    Noise model with single term for _all_ contributions of the model error.
    The difference to `SingleSensorNoise` is that each model error is assumed
    to be just a vector instead of a dict with sensor key.
    """

    def vector_contribution(self, raw_me):
        vector_terms = []
        for exp_me in raw_me.values():
            if isinstance(exp_me, dict):
                raise RuntimeError(
                    "The `SingleNoise` model assumes that your model error "
                    "returns just a list of numbers (e.g. numpy array), "
                    "yours returned a dict. Use `SingleSensorNoise` instead."
                )
            vector_terms.append(exp_me)
        return np.concatenate(vector_terms)


class UncorrelatedNoiseTerm(NoiseModelInterface):
    """
    Uncorrelated noise term that allows to specify exactly which output 
    (defined by key and sensor) from the model error is taken for the term.
    """

    def __init__(self):
        super().__init__()
        self.parameter_list.define("precision")
        self.terms = []

    def add(self, sensor, key=None):
        self.terms.append((sensor, key))

    def vector_contribution(self, raw_me):
        vector_terms = []
        for (sensor, key) in self.terms:
            vector_terms.append(raw_me[key][sensor])
        return np.concatenate(vector_terms)

    def loglike_contribution(self, raw_me):
        error = self.vector_contribution(raw_me)
        sigma = 1.0 / self.parameter_list["precision"] ** 0.5
        return self._loglike_term(error, sigma)


class UncorrelatedSensorNoise(NoiseModelInterface):
    """
    Uncorrelated noise term that allows to specify exactly which sensors
    from the model error are taken for the term.
    """

    def __init__(self, sensors):
        super().__init__()
        self.parameter_list.define("precision")
        self.sensors = sensors
        # self.sensors = []

    def vector_contribution(self, raw_me):
        vector_terms = []
        for exp_me in raw_me.values():
            for sensor, values in exp_me.items():
                if sensor in self.sensors:
                    vector_terms.append(values)

        if not vector_terms:
            raise RuntimeError(
                "The model error response did not contain any "
                f"contributions from sensors {[s.name for s in self.sensors]}."
            )

        return np.concatenate(vector_terms)

    def loglike_contribution(self, raw_me):
        error = self.vector_contribution(raw_me)
        sigma = 1.0 / self.parameter_list["precision"] ** 0.5
        return self._loglike_term(error, sigma)

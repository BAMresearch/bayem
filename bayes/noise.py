# imports from standard library
import math

# third party imports
import numpy as np

# local imports
from .parameters import ParameterList

"""
General information for understanding the noise classes
-------------------------------------------------------

The inference problem provides:
    model_error_dict:
        {model_error_key: {sensor: vector of model_error}}
    jacobian_dict:
        {model_error_key: {sensor: {parameters : vector/matrix of jacobian}}}

Both the log-likelihood function and VB require an input that is sorted by a
'noise group'. The conversion (basically a concatenation) of various pairs of
(model_error_key, sensor) is done via the 'NoiseModelInterface'.

See test/test_noise.py for an example and further explanation.
"""


class NoiseModelInterface:
    def model_error_terms(self, model_error_dict):
        """
        Rearranges the (model_error_key, sensor) ordering of 'model_error_dict'
        into a single numpy vector. Note that this method has to be overwritten
        when a new noise model is implemented, see 'UncorrelatedNoiseModel'.
        """
        raise NotImplementedError("Implement me!")

    def jacobian_terms(self, jacobian_dict):
        """
        Rearranges the (model_error_key, sensor) ordering of 'jacobian_dict'
        into a single numpy matrix. Note that this method has to be overwritten
        when a new noise model is implemented, see 'UncorrelatedNoiseModel'.
        """
        raise NotImplementedError("Implement me!")

    def loglike_contribution(self, model_error_dict):
        """
        Rearranges the (model_error_key, sensor) ordering of 'model_error_dict'
        into a single numpy vector and calculates its contribution to a
        loglikelihood function.
        """
        raise NotImplementedError("Implement me!")


class UncorrelatedNoiseModel(NoiseModelInterface):
    """
    Base class for associating one or more sensors with a specific noise model.
    The errors in these sensors (i.e. deviations from the forward model) are
    assumed to be independent from each other. Note that this class is most
    conveniently used in practise in form of one of the child classes defined
    afterwards (see UncorrelatedSensorNoise, UncorrelatedSingleNoise).
    """

    def __init__(self):
        # this error model is assumed to be Gaussian with zero mean; the
        # variance is defined by the precision parameter which is defined as
        # the inverse of the variance, i.e. precision = 1 / sigma**2, with
        # sigma being the standard deviation; so this model has one parameter
        # which is added to the noise model's parameter list
        self.parameter_list = ParameterList()
        self.parameter_list.define("precision")

        # _key_pairs is a list of pairs
        self._key_pairs = []

    def add(self, model_error_key, sensor):
        """
        Adds a ('model_error_key', 'sensor') pair to the noise model such that 
        the corresponding output, e.g. model_error_dict[model_error_key][sensor]
        or jacobian_dict[model_error_key][sensor] is added to the
        'self.model_error_terms' or 'self.jacobian_terms', respectively.

        Parameters
        ----------
        model_error_key : string
            States and experiments/measurements the given sensor relates to.
        sensor : dict
            Contains names of sensors as keys and the respective model error as
            a numpy-array as values.
        """
        self._key_pairs.append((model_error_key, sensor))

    def _define_key_pairs(self, model_error_dict):
        """
        Can be overwritten to automatically define 'self._key_pairs' for certain
        special cases. See UncorrelatedSensorNoise or UncorrelatedSingleNoise
        below.
        """
        pass

    def model_error_terms(self, model_error_dict):
        """
        A mask for calling self._by.noise for the case of the dict_of_dict (see
        the docstring of self._by_noise) being a model_error_dict.

        Parameters
        ----------
        model_error_dict : dict
            The keys specify different experiments/measurements, while the
            values are dictionaries with sensor names as keys and their
            model error as values, e.g. {'exp1' : {ForceSensor1 : [-0.3]}}.

        Returns
        -------
        list
            A list of numpy arrays containing the extracted values.
        """
        return self._by_noise(model_error_dict)

    def jacobian_terms(self, jacobian_dict):
        """
        A mask for calling self._by.noise for the case of the dict_of_dict (see
        the docstring of self._by_noise) being a jacobian_dict.

        Parameters
        ----------
        jacobian_dict : dict
            The keys specify different experiments/measurements, while the
            values are dictionaries with sensor names as keys and their
            derivatives, e.g. {'exp1' : {ForceSensor1 : [-0.3]}}.

        Returns
        -------
        list
            A list of numpy arrays containing the extracted values.
        """
        return self._by_noise(jacobian_dict)

    def loglike_contribution(self, model_error_dict):
        """
        Evaluates the log-likelihood for the sensors given in the given
        model_error_dict, assuming a Gaussian zero-mean distribution.

        Parameters
        ----------
        model_error_dict : dict
            The keys specify different experiments/measurements, while the
            values are dictionaries with sensor names as keys and their
            model error as values, e.g. {'exp1' : {ForceSensor1 : [-0.3]}}.

        Returns
        -------
        ll : float
            The computed value of the log-likelihood function.
        """
        # extract the numeric values from the given dictionary
        terms = self.model_error_terms(model_error_dict)
        # this is the noise model's only parameter
        prec = self.parameter_list["precision"]
        ll = 0.0
        for error in terms:
            # evaluate the Gaussian log-PDF with zero mean and a variance of
            # 1/prec for each error term and sum them up
            ll -= len(error) / 2 * math.log(2 * math.pi / prec)
            ll -= 0.5 * prec * np.sum(np.square(error))
        return ll

    def _by_noise(self, dict_of_dicts):
        """
        Extracts numeric values from 'dict_of_dicts' (which can be either
        'model_error_dict' or 'jacobian_dict') and concatenates them to a list
        of numpy vectors/matrices.

        Parameters
        ----------
        dict_of_dicts : dict
            The keys specify different experiments/measurements, while the
            values are dictionaries with sensor names as keys and their model
            error/derivative as values, e.g. {'exp1' : {ForceSensor1 : [-0.3]}}.

        Returns
        -------
        terms : list
            A list of numpy arrays containing the extracted values.
        """
        # this prepares self._key_pairs for the next step
        self._define_key_pairs(dict_of_dicts)
        # extracts the numeric values from dict_of_dicts and assembles them in
        # the list 'terms'; note that terms is a list of numpy arrays, e.g.
        # [np.array([0.3]), np.array([0.02]), np.array([-0.13])]
        terms = [dict_of_dicts[key][sensor]
                 for (key, sensor) in self._key_pairs]
        return terms

    def by_keys(self, terms):
        """
        Reverses the operation of 'self._by_noise'. Takes a list of lists with
        numeric values and maps them to the respective sensors.

        Parameters
        ----------
        terms : list
            A list of numpy arrays with the extracted values by self._by_noise.

        Returns
        -------
        dict_of_dicts : dict
            The keys specify different experiments/measurements, while the
            values are dictionaries with sensor names as keys and their model
            error/derivative as values, e.g. {'exp1' : {ForceSensor1 : [-0.3]}}.
        """

        # check if dictionary and list have the same number of elements
        if len(self._key_pairs) != len(terms):
            raise RuntimeError(
                f"Dimension mismatch: The noise model has "
                f"{len(self._key_pairs)} terms, you provided {len(terms)}."
            )

        # do the mapping assuming similar ordering
        split = {}
        for term, (key, sensor) in zip(terms, self._key_pairs):
            if key not in split:
                split[key] = {}
            split[key][sensor] = term

        return split


class UncorrelatedSingleNoise(UncorrelatedNoiseModel):
    """
    Noise model which applies to all sensors of the problem (i.e. all sensors
    which are defined in model_error_dict, see below). The errors in these
    sensors (i.e. deviations from the forward model) are assumed to be
    independent from each other.
    """

    def __init__(self):
        super().__init__()

    def _define_key_pairs(self, model_error_dict):
        """
        Writes all model errors from model_error_dict to the noise model. Note
        that this method overwrites the respective method from the super-class.

        Parameters
        ----------
        model_error_dict : dict
            The keys specify different experiments/measurements, while the
            values are dictionaries with sensor names as keys and their
            model error as values, e.g. {'exp1' : {ForceSensor1 : [-0.3]}}.
        """

        # do not do anything if self._key_pairs has already been defined
        # (possibly in a previous call of this function)
        if self._key_pairs:
            return

        # add all sensors in model_error_dict to the noise model
        for me_key, me in model_error_dict.items():
            for sensor in me:
                self.add(me_key, sensor)


class UncorrelatedSensorNoise(UncorrelatedNoiseModel):
    """
    Noise model which applies to all sensors stated in 'sensors' during
    initialization. The errors in these different sensors (i.e. deviations from
    the forward model) are assumed to be independent from each other.
    """

    def __init__(self, sensors):
        """
        Parameters
        ----------
        sensors : string, list or tuple of strings
            One or more strings specifying the sensors to be described by this
            noise model, e.g. ['ForceSensor1', 'ForceSensor2']
        """
        super().__init__()
        # sensor should be written to a list-like attribute, so that one can
        # check if a given string is in that list
        if type(sensors) is list or type(sensors) is tuple:
            self.sensors = sensors
        else:  # this is the case when sensors is a single string
            self.sensors = [sensors]

    def _define_key_pairs(self, model_error_dict):
        """
        Extracts the model errors from model_error_dict for the sensors that
        have been stated in 'sensors' during initialization. Note that this
        method overwrites the respective method from the super-class.

        Parameters
        ----------
        model_error_dict : dict
            The keys specify different experiments/measurements, while the
            values are dictionaries with sensor names as keys and their
            model error as values, e.g. {'exp1' : {ForceSensor1 : [-0.3]}}.
        """

        # do not do anything if self._key_pairs has already been defined
        # (possibly in a previous call of this function)
        if self._key_pairs:
            return

        # add only those sensors from the given model_error_dict to the noise
        # model which have been stated in sensors during initialization
        for me_key, me in model_error_dict.items():
            for sensor in me:
                if sensor in self.sensors:
                    self.add(me_key, sensor)

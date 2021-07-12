# standard library imports
from collections import OrderedDict

# third party imports
import numpy as np

# local imports
from .latent import LatentParameters


class InferenceProblem:
    """
    An InferenceProblem is defined by a set of parameters to calibrate (called
    latent parameters), a method to compute the model error (i.e. the difference
    between the computed model values and the experimentally measured values)
    and a noise model which translates the model error into an evaluation of
    a likelihood function. Note that this class is a base class to be extended
    by the specific problem or the specific methodology at hand (see for example
    the class VariationalBayesProblem in vb.py).
    """

    def __init__(self):
        self.latent = LatentParameters()  # name : value
        self.model_errors = OrderedDict()  # key : model error object
        self._noise_models = OrderedDict()  # key : noise model object

    def add_model_error(self, model_error, key=None):
        """
        Adds a model_error to the InferenceProblem definition.

        Parameters
        ----------
        model_error : object
            An instance of ModelErrorInterface, see model_error.py
        key : string, optional
            A name for this model error, for example the name of the experiment
            this model error was derived from. If no key is given, it is defined
            as the number of its definition. For example if it is the second
            error_model defined, its key will be 2.

        Returns
        -------
        key : int
            The key the given model_error can be accessed with from the
            model_error dictionary.
        """
        # define the key and assert that it is not taken yet
        key = key or len(self.model_errors)
        assert key not in self.model_errors
        # add the model error to the dictionary under the given/derived key
        self.model_errors[key] = model_error
        return key

    def add_noise_model(self, noise_model, key=None):
        """
        Adds a noise_model to the InferenceProblem definition.

        Parameters
        ----------
        noise_model : object
            An instance of NoiseModelInterface, see noise.py
        key : string, optional
            A name for this noise model, for example 'PositionSensorNoise'. If
            no key is given, it is generically defined with the number of its
            definition. For example if it is the second noise_model defined, its
            key will be 'noise2'.

        Returns
        -------
        key : int
            The key the given noise_model can be accessed with from the
            noise_model dictionary.
        """
        # define the key and assert that it is not taken yet
        key = key or f"noise{len(self._noise_models)}"
        assert key not in self._noise_models
        # add the noise model to the dictionary under the given/derived key
        self._noise_models[key] = noise_model
        return key

    @property
    def noise_models(self):
        """
        Returns the currently defined noise models.

        Returns
        -------
        OrderedDict
            The currently defined noise models as an ordered dictionary with
            their names as keys and the corresponding objects as values.
        """
        if not self._noise_models:
            raise RuntimeError(
                "No noise model has been defined yet! In order to define one " +
                "see bayes.noise for options and then call .add_noise_model " +
                "to add it to the inference problem."
            )
        return self._noise_models

    def define_shared_latent_parameter_by_name(self, name):
        """
        If the same parameter appears in multiple model error definitions one
        can define all of them as the same latent parameter with this method.

        Parameters
        ----------
        name : string
            The name of the shared latent parameter.
        """
        # loop over all model error objects and check if their parameters
        # contain the specified parameter with name 'name'
        for model_error in self.model_errors.values():
            # check for the parameter_list attribute
            if not hasattr(model_error, 'parameter_list'):
                raise AttributeError(
                    "This method requires all 'model_error' objects to have " +
                    "a 'parameter_list' attribute. At least one of them does" +
                    "not have this attribute."
                )
            # add the model_error's parameter_list to the problem's latent
            # parameters if it contains the specified parameter
            if name in model_error.parameter_list:
                self.latent[name].add(model_error.parameter_list, name)

    def __call__(self, number_vector):
        """
        Updates the problems latent parameters and re-evaluates each model error
        with these updated latent parameters.

        Parameters
        ----------
        number_vector : array_like
            A numeric 1D-vector containing values to set for latent parameters.

        Returns
        -------
        result : dict
            Essentially a copy of self.model_errors with re-evaluated numeric
            values for the model errors for the updated latent parameters.
        """
        # update the latent parameters
        self.latent.update(number_vector)
        # re-evaluate each model error with updated latent parameters
        result = {}
        for key, me in self.model_errors.items():
            result[key] = me()
        return result

    def loglike(self, number_vector):
        """
        Evaluate the log-likelihood function over all defined noise models.

        Parameters
        ----------
        number_vector : array_like
            A numeric 1D-vector containing values to set for latent parameters.

        Returns
        -------
        ll : float
            The computed value of the log-likelihood function.
        """

        # channel the numbers given in number_vector to the corresponding
        # parameters to update the latent parameters accordingly
        self.latent.update(number_vector)

        # loop over each given error model object and call it to compute the
        # model error with the updated latent parameters
        raw_me = {}
        for key, me in self.model_errors.items():
            raw_me[key] = me()

        # the updated model errors can now be fed to the noise models in order
        # to sum up their contributions to the log-likelihood value
        ll = 0.0
        for noise_key, noise_term in self.noise_models.items():
            ll += noise_term.loglike_contribution(raw_me)

        return ll

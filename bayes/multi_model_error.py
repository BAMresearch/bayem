from bayes.parameters import *
from collections import OrderedDict
import numpy as np


class MultiModelError:
    """
    Purpose:

        * Join a different individual model errors (each with a __call__ routine
          to determine the individual model error) in a joint interfaces via a __call__
          method that can e.g. be provided to an optimization procedure such as scipy-optimize
        * Enable the usage of shared variables that are identical and multiple model errors
          and are thus only once in the global optimization.

    Example:
        * Join multiple data sets (and individual model errors) of a tensile test and potentially
          other tests such as a three point bending test to determine joint material parameters
          such as the Youngs modulus.

    Idea:

        * Idea: The individual model errors as well as the paramter lists are stored a dictionary
          with a unique key to indentify the indivual model errors. The same key is used in the
          joint_parameter_list to store the complete parameter list required as input
          to compute the model error. This joint_parameter_list that provides the functionality
          to update identical variables (labeled shared) in multiple parameter lists.
    """

    def __init__(self):
        self.mes = {}
        self.keys = []

        self.n = 0

        self.latent = LatentParameters()
        self.shapes = {}

    def add(self, model_error, parameters, key=None):
        """
        Add an individual model error.
        model_error:
            local model error that provides an __call__ method to
            compute the model error

        parameters:
            local list of parameters (see ModelParameter in parameters.py)
            required to compute the model error (in the model error, this list
            is usually passed to the forward model)
        key:
            the key used in the dictionary to access the local model_error and local
            parameter_list. If not provided an integer key is used.
        """
        key = key or self.n
        assert key not in self.keys
        self.n += 1

        self.mes[key] = model_error
        self.keys.append(key)
        self.latent.define_model_parameters(parameters, key)

        return key

    def __call__(self, parameter_vector):
        """
        Updates all latent parameters in the joint_parameter_list
        based on the parameter_vector and evaluates each individual model error
        and concatenates the result into a long vector
        parameter_vector:
            "global" parameter vector exposed to the joint optimization.
            The dimension must be identical to the number of latent variables
            (shared variables are only a single latent variable)
        """

        updated_parameters = self.latent.update(parameter_vector)
        result = []
        for key in self.keys:
            prm = updated_parameters[key]
            me = self.mes[key]
            single_model_error = me(prm)

            self.shapes[key] = len(single_model_error)
            result.append(single_model_error)
        return np.concatenate(result)

    def evaluate(self, parameter_vector):
        """
        Updates all latent parameters in the joint_parameter_list
        based on the parameter_vector and evaluates each individual model error
        as opposed to the call function, this is not concatenated,
        but stored in separate dictionarys with the key being the model key:
            "global" parameter vector exposed to the joint optimization.
            The dimension must be identical to the number of latent variables
            (shared variables are only a single latent variable)
        """
        updated_parameters = self.latent.update(parameter_vector, return_copy=False)
        result = OrderedDict()
        for key in self.keys:
            prm = updated_parameters[key]
            me = self.mes[key]
            single_model_error = me.evaluate(prm)

            self.shapes[key] = len(single_model_error)
            result[key] = single_model_error
        return result

    def split_by_key(self, array):
        """
        splits the complete array and extracts the individual components
        result of __call__ and split_by_key is essentially similar to directly calling evaluate
        """
        array_by_key = {}
        offset = 0
        for key in self.keys:
            l = self.shapes[key]
            array_by_key[key] = array[offset : offset + l]
            offset += l
        return array_by_key

    def uncorrelated_normal_prior(self):
        """
        Joins all individual parameter lists and creates a normal prior for all latent variables

        shared:
            dictionary of variables to be shared (only used once in the global problem)
        """
        return UncorrelatedNormalPrior(self.latent)

    def build_joint_noise_pattern(self):
        offset = 0
        self.noise_pattern = []
        for key, me in self.mes.items():
            noise_pattern = me.noise_pattern()

            len_p = []

            for p in noise_pattern:
                self.noise_pattern.append(p + offset)
                len_p.append(max(p) if len(p) != 0 else 0)

            offset += max(len_p) + 1
        pass

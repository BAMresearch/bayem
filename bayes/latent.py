import collections
import copy
import logging
from typing import Dict, Hashable, List, Tuple, NamedTuple

import numpy as np  # type: ignore
from tabulate import tabulate

from bayes.parameters import ParameterList

logger = logging.getLogger(__name__)


class LatentParameters(collections.OrderedDict):
    """
    The models (can be forward models, noise models) each require a named 
    `ParameterList` as input, where some of those named parameters have:
        1) different names 
                (e.g. "Young's modulus" in model1 and "E" in model2)
           but should be infered simultaneously.  
        2) same names 
                (e.g. "T" for time in model1 and "T" for temperature in model2)
            but should be infered seperately. 

    Thus, we refer to the parameters in the individual models as `local_name`
    and the latent variable to be infered as `global_name`. The `LatentParameters`
    class maps between those. In the example above, we could have the 
    following mapping:

            global name      length  model model_key    local name
            -------------  --------  -----------  ---------------
            E                     1  model1       Young's modulus
            E                     1  model2       E
            time                  1  model1       T
            temperature           1  model2       T

    Basically for checking if a model or a model model_key exists, we need to pass
    the available `models` to the `LatentParameters`.

        > l = LatentParameters(models={"model1": model1, "model2":model2})

    The indented use to generate these mappings is now to access the 
    `global_name` from the `LatentParameters` via dict-access and call the
    `add` method with the model model_key or the model itself and the local name.
    
        > l["E"].add(model1, "Young's modulus")
        > l["E"].add(model2, "E")
        > l["time"].add(model1, "T")
        > l["temperature"].add(model2, "T")

    We now have defined 3 global latent parameters and the main usage is to call

        > new_prm_lists = l.updated_parameters([1, 2, 3])

    which will result in two (one for each model) `ParameterLists` as

            model1 : {"Young's modulus": 1, 'T': 2}
            model2 : {'E': 1, 'T': 3}

    that can then be passed to the models for their evaluation.
    """

    def __init__(self):
        """
        We have to keep track of the `self._models` to check, if the model
        the user is referring to actually exists. Also the model may have
        a `get_shape(local_name)` method that returns the length > 1 for
        vector valued parameters.
        """
        super().__init__()
        self._models = {}

    def __missing__(self, global_name):
        """
        This method is called, whenever a `global_name` is accessed that is 
        not yet part of the `LatentParameters`. This is the bit of magic that
        allows us to call

          > latent_parameters["A"].add(...) # __missing__("A") is called!
          > latent_parameters["A"].add(...) # __getitem__("A") is called.

        As the `ParameterListReferences.add` needs to know the `models` (to 
        get the parameter length and perform checks) and the `global_name`
        (for more meaningful debug messages), we pass it.
        """
        self[global_name] = ParameterListReferences(self._models, global_name)
        return self[global_name]

    def add_model(self, model_key, model):
        self._models[model_key] = model

    @property
    def vector_length(self):
        return sum(l.N for l in self.values())

    def updated_parameters(
        self, number_vector: np.ndarray
    ) -> Dict[Hashable, ParameterList]:
        """
        Returns a dict {model_key: ParameterList} with the values from 
        `number_vector`. See the example above for details.
        """

        if self.vector_length != len(number_vector):
            msg = f"These latent parameters have a total length of "
            msg += f"{self.vector_length}, you provided a vector of length "
            msg += f"{len(number_vector)}"
            raise InconsistentLengthException(msg)

        start_idx = 0

        lists = {}
        for global_name, latent in self.items():

            if latent.N == 1:
                value = number_vector[start_idx]
            else:
                value = number_vector[start_idx : start_idx + latent.N]

            for model_key, local_name in latent:
                if model_key not in lists:
                    lists[model_key] = ParameterList()
                lists[model_key].define(local_name, value)

            start_idx += latent.N

        return lists

    def global_name(self, model_key, local_name):
        """
        Find the `global_name` for a given `(model_key, local_name)` entry.
        """
        for global_name, latent in self.items():
            if (model_key, local_name) in latent:
                return global_name
        raise RuntimeError(
            f"{local_name} has no global_name, since it is not defined as latent!"
        )

    def global_indices(self, global_name):
        """
        Returns a list of indices associated to the `global_name`. In case of 
        only scalar parameters, the ith latent parameters has 
            global_indices = [i]
        but vector valued parameters may have more.
        """
        start_idx = 0
        for _global_name, _latent in self.items():
            if global_name == _global_name:
                return list(range(start_idx, start_idx + _latent.N))
            start_idx += _latent.N
        raise RuntimeError(f"{global_name} is not defined as latent!")

    def __str__(self) -> str:
        """
        Prints the global-local parameter mapping as a fancy table
        """
        to_print = []
        for global_name, latent in self.items():
            for model_key, local_name in latent:
                to_print.append((global_name, latent.N, model_key, local_name))

        return tabulate(
            to_print, headers=["global name", "length", "model model_key", "local name"]
        )

    def check_priors(self):
        """
        Checks, if there is a prior assigned to each latent parameter, aka 
        `global_name`.
        """
        for global_name, latent in self.items():
            if latent.prior is None:
                raise RuntimeError(
                    f"You defined {global_name} as latent but did not provide a prior distribution!."
                )


class InconsistentLengthException(Exception):
    pass


class ParameterListReferences(list):
    """
    A combination of ``

    """

    def __init__(self, models, global_name):
        self.N = None
        self._models = models
        self._global_name = global_name
        self.__prior = None

    def _find_model(self, model_or_key):
        for known_key, known_model in self._models.items():
            if model_or_key == known_key or model_or_key == known_model:
                return known_key, known_model
        raise RuntimeError("AAAAH!")

    @property
    def prior(self):
        return self.__prior

    @prior.setter
    def prior(self, new_prior):
        if self.__prior is not None:
            msg = f"Prior for {self._global_name} was already set to '{self.prior}' "
            msg += f"and is now overwritten with '{new_prior}'."
            raise RuntimeError(msg)

        self.__prior = new_prior

    def add_shared(self, prior=None):
        """
        Defines `global_name` as a latent parameter in all models
        """
        for model_key in self._models:
            self.add(model_key, self._global_name)
        self.prior = prior

    def add(self, model_or_key, local_name=None):
        local_name = local_name or self._global_name
        model_key, model = self._find_model(model_or_key)

        try:
            N = model.get_shape(local_name)
        except AttributeError:
            N = 1

        if self.N is None:
            self.N = N
        else:
            if self.N != N:
                raise InconsistentLengthException("TODO: some info")

        parameter_reference = (model_key, local_name)
        if parameter_reference in self:
            msg = f"{parameter_reference} is already associated to the global "
            msg += f"parameter {self._global_name}. No need to add it again!"
            logger.warning(msg)
        self.append(parameter_reference)


if __name__ == "__main__":
    model1 = "this does not"
    model2 = "matter here."

    l = LatentParameters(models={"model1": model1, "model2": model2})
    l["E"].add(model1, "Young's modulus")
    l["E"].add(model2, "E")
    l["time"].add(model1, "T")
    l["temperature"].add(model2, "T")

    print(l)

    new_prm_lists = l.updated_parameters([1, 2, 3])
    for model_key, prm_list in new_prm_lists.items():
        print(model_key, ":", prm_list.p)

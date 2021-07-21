import copy
import collections
from typing import Hashable
import logging

import numpy as np
from tabulate import tabulate

from .parameters import ParameterList


logger = logging.getLogger(__name__)


class LatentParameter(list):
    def __init__(self, N):
        self.N = N


class InconsistentLengthException(Exception):
    pass


class LatentParameters(collections.OrderedDict):
    def __init__(self):
        super().__init__()
        # This member is just for convenience/performance such that the
        # `updated_parameters` do not have to build it again and can
        # just copy it.
        self._empty_parameter_lists: dict[Hashable, ParameterList] = {}

    def add(
        self, global_name: str, local_name: str, model_error_key: Hashable, N: int = 1
    ) -> None:
        self._empty_parameter_lists[model_error_key] = ParameterList()

        if global_name not in self:
            self[global_name] = LatentParameter(N)

        latent = self[global_name]
        if latent.N != N:
            raise InconsistentLengthException("TODO: some info")

        parameter_reference = (model_error_key, local_name)
        if parameter_reference in latent:
            msg = f"{parameter_reference} is already associated to the global "
            msg += f"parameter {global_name}. No need to add it again!"
            logger.warning(msg)
        latent.append((model_error_key, local_name))

    def updated_parameters(
        self, number_vector: np.ndarray
    ) -> dict[Hashable, ParameterList]:

        # maybe make that a member?
        current_length = sum(l.N for l in self.values())

        if current_length != len(number_vector):
            msg = f"These latent parameters have a total length of "
            msg += f"{current_length}, you provided a vector of length "
            msg += f"{len(number_vector)}"
            raise InconsistentLengthException(msg)

        lists = copy.deepcopy(self._empty_parameter_lists)

        start_idx = 0

        for global_name, latent in self.items():

            if latent.N == 1:
                value = number_vector[start_idx]
            else:
                value = number_vector[start_idx : start_idx + latent.N]

            for model_error_key, local_name in latent:
                lists[model_error_key].define(local_name, value)
            start_idx += latent.N

        return lists

    def global_name(self, local_name):
        for global_name, latent in self.items():
            for _, ref_local_name in latent:
                if ref_local_name == local_name:
                    return global_name
        return None 

    def global_indices(self, global_name):
        start_idx = 0
        for _global_name, _latent in self.items():
            if global_name == _global_name:
                return list(range(start_idx, start_idx + _latent.N))
            start_idx += _latent.N
        assert False


    def __str__(self):
        to_print = []
        for global_name, latent in self.items():
            for model_error_key, local_name in latent:
                to_print.append((global_name, latent.N, model_error_key, local_name))

        return tabulate(
            to_print, headers=["global name", "length", "model error", "local name"]
        )

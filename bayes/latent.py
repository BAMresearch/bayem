from collections import OrderedDict
from operator import itemgetter


class LatentParameter(list):
    def __init__(self, latent_parameters):
        # we first need an instance of the global list to update the indices!
        self._latent_parameters = latent_parameters

        self.N = None
        self.start_idx = None

    def add(self, parameter_list, parameter_name):
        assert parameter_name in parameter_list
        assert not self.has(parameter_list, parameter_name)

        try:
            N = len(parameter_list[parameter_name])
        except TypeError:  # "has no __len__"
            N = 1

        if self.N is None:
            self.N = N
            self._update_idx()
        else:
            assert self.N == N

        self.append((parameter_list, parameter_name))

    def _update_idx(self):
        length = 0
        for key, latent in self._latent_parameters.items():
            latent.start_idx = length
            length += latent.N

    def global_index_range(self):
        return list(range(self.start_idx, self.start_idx + self.N))

    def update(self, number_vector):
        values = itemgetter(*self.global_index_range())(number_vector)
        for parameter_list, parameter_name in self:
            parameter_list[parameter_name] = values

    def has(self, parameter_list, parameter_name):
        return (parameter_list, parameter_name) in self


class LatentParameters(OrderedDict):
    def update(self, number_vector):
        for latent in self.values():
            latent.update(number_vector)

    def __missing__(self, latent_name):
        """
        This method is called whenever you access a new latent parameter.
        """
        self[latent_name] = LatentParameter(self)
        return self[latent_name]

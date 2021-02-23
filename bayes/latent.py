from collections import OrderedDict
from operator import itemgetter


class LatentParameter(list):
    """
    Represents a single latent parameter that is mapped to one or more
    individual (ParameterList, name) pairs stored in this list.
    """

    def __init__(self, latent_parameters):
        # we first need an instance of the global list to update the indices!
        self._latent_parameters = latent_parameters

        self.N = None  # number of (scalar) parameters of this
        self.start_idx = None  # start index in the

    def add(self, parameter_list, parameter_name):
        """
        Adds `parameter_name` of `parameter_list` to this LatentParameter. So
        if we change the value of this LatentParameter in `update`,
        `parameter_name` of `parameter_list` will be changed as well.
        """
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
        """
        As soon as a new latent parameter is added, the start indices of the
        existing parameters must be updated. 
        """
        length = 0
        for key, latent in self._latent_parameters.items():
            latent.start_idx = length
            length += latent.N

    def global_index_range(self):
        """
        Returns a list of global vector indices associated with this 
        LatentParameter.

        Example:
            The parameters "A" and "B" of parameter list
                p["A"] = [6., 1., 7., 4.]
                p["B"] = 42.
            are added via `self.add`.

            Thus, the _global vector_ of latent variables has length 5 and 
            could look like: [A, B, B, B, B].

            This means:
                latent["A"].global_index_range() == [0, 1, 2, 3]
                latent["B"].global_index_range() == [4]

        Remark:
            At some point, e.g. for defining prior distributions for latent
            parameters, the user can use this method to define individual 
            distributions like:

            for latent_parameter in latent.values():
                for index in latent_parameter.global_index_range():
                    prior[index] = SomeScalarDistribution()

            Alternatively, vector valued distributions could be defined like:

            for latent_parameter in latent.values():
                prior.append(MultivariateDistribution(length=latent_parameter.N))

        """
        return list(range(self.start_idx, self.start_idx + self.N))

    def update(self, number_vector):
        """
        Extracts the `global_index_range` from `number_vector` and assigns it
        to all (parameter_list,parameter_name) pairs in `self`.
        """
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
        This method is called whenever you access a new latent parameter,
        mainly resulting in an `DefaultDict`.

        To update the global start indices when parameters are added, we pass
        `self` - so the LatentParameters - to the newly created
        LatentParameter.
        """
        self[latent_name] = LatentParameter(self)
        return self[latent_name]

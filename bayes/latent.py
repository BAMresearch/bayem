from collections import OrderedDict
from operator import itemgetter

def len_or_one(vector_or_scalar):
    """
    Returns the length of a vector or 1 for a scalar
    """
    try:
        return len(vector_or_scalar)
    except TypeError:  # "has no __len__"
        return 1

class LatentParameter(list):
    """
    Represents a single latent parameter that is mapped to one or more
    individual (ParameterList, name) pairs stored in this list.
    """

    def __init__(self, latent_parameters, name):
        # we first need an instance of the global list to update the indices!
        self._latent_parameters = latent_parameters
        self._name = name  # for better RuntimeErrors

        self.N = None  # number of (scalar) parameters of this
        self.start_idx = None  # start index in the

    def add(self, parameter_list, parameter_name):
        """
        Adds `parameter_name` of `parameter_list` to this LatentParameter. So
        if we change the value of this LatentParameter in `update`,
        `parameter_name` of `parameter_list` will be changed as well.
        """
        if parameter_name not in parameter_list:
            raise RuntimeError(
                f"Parameter {parameter_name} is not part of {parameter_list},"
                f"so it cannot be added to the latent variable {self._name}."
            )

        if self.has(parameter_list, parameter_name):
            raise RuntimeError(
                f"Parameter {parameter_name} of {parameter_list} is already"
                f"associated with latent parameter {self._name}!"
            )

        N = len_or_one(parameter_list[parameter_name])
        
        if self.N is None:
            self.N = N
            self._update_idx()
        else:
            if self.N != N:
                raise RuntimeError(
                    f"The latent parameter {self._name} is defined with length"
                    f"{self.N}. This does not match length {N} of parameter"
                    f"of {parameter_name} in {parameter_list}!"
                )

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
        values = self.values(number_vector)
        for parameter_list, parameter_name in self:
            parameter_list[parameter_name] = values
    
    def set_value(self, value):
        """
        Updates the parameters associated to self in all parameter lists.
        Checks, if the dimensions match.
        """
        assert self.N == len_or_one(value)

        for parameter_list, parameter_name in self:
            parameter_list[parameter_name] = value
    

    def values(self, number_vector):
        return itemgetter(*self.global_index_range())(number_vector)

    def has(self, parameter_list, parameter_name):
        return (parameter_list, parameter_name) in self

    def value(self, number_vector=None):
        """
        Returns the value this latent parameter either from (one of) its 
        parameter list(s) or from `number_vector`, if provided.
        """
        if number_vector is None:
            if not self:  # = self is empty
                raise RuntimeError(f"There is no parameter associated to {self._name}!")

            parameter_list, parameter_name = self[0]
            return parameter_list[parameter_name]

        else:
            return self.values(number_vector)


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
        self[latent_name] = LatentParameter(self, latent_name)
        return self[latent_name]

    def __str__(self):
        l_max = max(len(name) for name in self)
        s = ""
        for latent_name, latent in self.items():
            s += f"{latent_name:{l_max}} = {latent.value()}\n"
        return s

    def latent_names(self, parameter_list):
        """
        Returns the latent parameter names of `parameter_list`.
        """
        names = []
        for global_name, latent in self.items():
            for (prm_list, local_name) in latent:
                if prm_list == parameter_list:
                    names.append((local_name, global_name))
        return names

# imports from standard library
from collections import OrderedDict
from operator import itemgetter

# local imports
from .subroutines import len_or_one


class LatentParameter(list):
    """
    Represents a single latent parameter that is mapped to one or more
    individual (ParameterList, name) pairs stored in this list.
    """

    def __init__(self, latent_parameters, name):
        """
        Parameters
        ----------
        latent_parameters : LatentParameters object
            Dictionary-like mapping between parameter names and values
        name : string
            The name of the parameter (must appear as key in parameter_list)
        """

        # initialize super-class
        super().__init__()

        # we need an instance of the global list to update the indices!
        self._latent_parameters = latent_parameters
        self._name = name  # the global name of this latent parameter

        # to be set later
        self.N = None  # number of scalar values defining this latent parameter
        self.start_idx = None  # start index in the global parameter vector

    def add(self, parameter_list, parameter_name):
        """
        Connects parameter_name as it appears in parameter_list with the
        LatentParameter object. So if we change the value of the LatentParameter
        object (using its update method) the value of parameter_name in
        parameter_list will be changed as well.

        Parameters
        ----------
        parameter_list : ParameterList object
            Dictionary-like mapping between parameter names and values
        parameter_name : string
            The name of the parameter (must appear as key in parameter_list)
        """

        # the given parameter_name must appear in the given parameter_list
        if parameter_name not in parameter_list:
            raise RuntimeError(
                f"Parameter {parameter_name} is not part of {parameter_list},"
                f"so it cannot be added to the latent variable {self._name}."
            )

        # the given pair of (parameter_list, parameter_name) must be unique
        if self.has(parameter_list, parameter_name):
            raise RuntimeError(
                f"Parameter {parameter_name} of {parameter_list} is already"
                f"associated with latent parameter {self._name}!"
            )

        # get the length/dimension of the parameter_name's value in the given
        # parameter_list, and check if it is consistent with the other ones
        N = len_or_one(parameter_list[parameter_name])
        if self.N is None:  # in this case the the given pair is the first one
            self.N = N
            self._update_idx()
        else:  # in this case self.N was defined by a previously added pair
            if self.N != N:  # hence both lengths must be the same
                raise RuntimeError(
                    f"The latent parameter {self._name} is defined with length"
                    f"{self.N}. This does not match length {N} of parameter"
                    f"of {parameter_name} in {parameter_list}!"
                )

        # if no conflicts are found, the given pair is added to the object
        self.append((parameter_list, parameter_name))

    def has(self, parameter_list, parameter_name):
        """Checks if the given pair already exists in the object."""
        return (parameter_list, parameter_name) in self

    def set_value(self, value):
        """
        Updates the parameters associated with the latent parameter.

        Parameters
        ----------
        value : array_like
            A numeric 1D-vector with values to set for this latent parameter.
        """
        # check dimensional consistency
        if self.N != len_or_one(value):
            raise RuntimeError(
                f"Dimension mismatch: Latent parameter '{self._name}' is ",
                f"of length {self.N} and you provided a vector of length ",
                f"{len_or_one(value)}!"
            )

        # update all associated parameters
        for parameter_list, parameter_name in self:
            parameter_list[parameter_name] = value

    def _update_idx(self):
        """
        Updates the start indices (referring to the numeric latent parameter
        vector called 'number_vector' in the code below) of the previously
        defined latent parameters when a new latent parameter is added.
        """
        length = 0
        for key, latent in self._latent_parameters.items():
            latent.start_idx = length
            length += latent.N

    def update(self, number_vector):
        """
        Updates all parameters identified with the latent parameter from the
        global latent parameter vector.

        Parameters
        ----------
        number_vector : array_like
            A numeric 1D-vector containing values to set for latent parameters.
        """
        self.set_value(self.values(number_vector))

    def values(self, number_vector):
        """
        Extracts the values corresponding to the latent parameter from the
        global latent parameter vector.

        Parameters
        ----------
        number_vector : array_like
            A numeric 1D-vector containing values to set for latent parameters.

        Returns
        -------
        tuple
            The values from number_vector that refer to the latent parameter
        """
        return itemgetter(*self.global_index_range())(number_vector)

    def global_index_range(self):
        """
        Returns a list of global vector indices (called number_vector in this
        script) associated with the latent parameter.

        Example:
        --------
        The parameters "A" and "B" of parameter list
            p["A"] = [6., 1., 7., 4.]
            p["B"] = 42.
        are added via 'self.add'.

        Thus, the _global vector_ of latent variables has length 5 and
        could look like: [A, B, B, B, B].

        This means:
            latent["A"].global_index_range() == [0, 1, 2, 3]
            latent["B"].global_index_range() == [4]

        Remark:
        -------
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

    def value(self, number_vector=None):
        """
        Returns the value this latent parameter either from (one of) its 
        parameter list(s) or from number_vector, if provided.

        Parameters
        ----------
        number_vector : array_like, optional
            A numeric 1D-vector containing values to set for latent parameters.

        Returns
        -------
        tuple
            The values from number_vector that refer to the latent parameter
        """
        if number_vector is None:
            if not self:  # meaning self (which is a list) is empty
                raise RuntimeError(
                    f"There is no parameter associated to {self._name}!"
                )
            # read the value from the first pair
            parameter_list, parameter_name = self[0]
            return parameter_list[parameter_name]
        else:
            return self.values(number_vector)

    def unambiguous_value(self):
        """
        Returns the value this latent parameter either from (one of) its 
        parameter list(s), checking that _all_ values are equal.
        
        Returns
        -------
        array_like
            The value of the latent parameter taken from the parameter_lists
            which was checked on consistency between all parameter_lists.
        """
        p_list0, p_name0 = self[0]
        value0 = p_list0[p_name0]
        for p_list, p_name in self:
            value = p_list[p_name]
            if value != value0:
                raise RuntimeError(
                    f"The values of the shared global latent parameter",
                    f"'{self._name}' differ in the individual parameter lists:",
                    f"\nParameter '{p_name0, value0}' in \n{p_list0} vs. ",
                    f"parameter '{p_name, value}' in \n{p_list}"
                )
        return value0


class LatentParameters(OrderedDict):
    """
    An ordered dictionary mapping latent parameter names to LatentParameter
    objects. It provides the capability to update all parameters identified as
    the latent parameters from a numeric vector (see the update method).
    """

    def update(self, number_vector):
        """"
        Update the values of all parameters identified as latent parameters
        according to the values provided in number_vector.

        Parameters
        ----------
        number_vector : array_like
            A numeric 1D-vector containing values to set for latent parameters.
        """
        # check for dimensional consistency
        n_parameters = self.vector_length
        if n_parameters != len(number_vector):
            raise RuntimeError(
                f"Dimension mismatch: The global latent parameter vector ",
                f"contains {n_parameters} values, but your vector contains ",
                f"{len(number_vector)} values!"
            )

        # update all latent parameters
        for latent in self.values():
            latent.update(number_vector)

    @property
    def vector_length(self):
        """
        Returns the number of scalar values needed to define the global latent
        parameter vector. If you had one 1D and one 3D latent parameter, the
        returned value would be 4.
        """
        return sum(latent_parameter.N for latent_parameter in self.values())

    def __missing__(self, latent_name):
        """
        This method is called whenever you access a new latent parameter,
        mainly resulting in an 'DefaultDict'. To update the global start indices
        when parameters are added, we pass 'self' - so the LatentParameters - to
        the newly created LatentParameter.
        
        Parameters
        ----------
        latent_name : string
            The name of the new latent variable.
            
        Returns
        -------
        LatentParameters-object
            The updated 'self' containing the new latent parameter.
        """
        self[latent_name] = LatentParameter(self, latent_name)
        return self[latent_name]

    def __str__(self):
        """
        Defines the printed output when passing an object of this class to the
        print() method.

        Returns
        -------
        string
            The name:value pairs of all latent parameters.
        """
        l_max = max(len(name) for name in self)
        s = ""
        for latent_name, latent in self.items():
            s += f"{latent_name:{l_max}} = {latent.value()}\n"
        return s

    def latent_names(self, parameter_list):
        """
        Returns the latent parameter names of 'parameter_list'.

        Parameters
        ----------
        parameter_list : ParameterList-object
            A parameter list possibly containing parameters that are identified
            as latent parameters.

        Returns
        -------
        list
            A list containing the names of the latent parameters found in
            the given parameter_list.
        """
        names = []
        for global_name, latent in self.items():
            for (prm_list, local_name) in latent:
                if prm_list == parameter_list:
                    names.append((local_name, global_name))
        return names

    def get_vector(self, overwrite=None):
        """
        Assembles the global vector of latent parameters from the values stated
        in the individual parameter lists. If the value(s) of a parameter are
        defined in the 'overwrite' dictionary, these values are used instead.

        Parameters
        ----------
        overwrite : dict
            May contain pairs like global parameter name : parameter value.
            Note that the dimension of the parameter value must match the
            dimension of the respective parameter.

        Returns
        -------
        v : list
            A global latent parameter vector.
        """

        # this prevents a mutable default argument
        if overwrite is None:
            overwrite = {}

        # assemble the global vector
        v = []
        for name, latent in self.items():
            if name in overwrite:
                value = overwrite[name]
                N_value = len_or_one(value)
                if latent.N != N_value:
                    raise RuntimeError(
                        f"Dimension mismatch: Latent parameter '{name}' has ",
                        f" length {latent.N}, but you provided the value ",
                        f"{value} which is of length {N_value}."
                    )
            else:  # extract values from the parameter_lists
                try:
                    value = latent.unambiguous_value()
                except RuntimeError as e:
                    msg_add = f"You can fix that by adding the dictionary " +\
                              f"entry '{name} : your_value' to the optional " +\
                              f"'overwrite' argument of this method."
                    raise RuntimeError(str(e) + msg_add)

            # add the obtained value to the global vector
            if latent.N == 1:
                v.append(value)
            else:
                for i in value:
                    v.append(i)

        return v

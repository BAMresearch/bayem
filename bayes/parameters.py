from collections import OrderedDict

"""
Purpose:

    * Map a _vector_ (just numbers) of latent parameters to meaningful "names"
      used in the models.
    * Deal with combination of models that possibly share parameters.

Example:

    * Two "CarPassesOverBridge" models (with different data sets) that
      share the unknown "YoungsModulus" parameter, but each have N unique 
      sensor related parameters, called "SensorOffset1..N". The total 
      number of parameters would then be 2N (unique) + 1 (shared). 

Idea: 

    * Idea: Introduce a "key" that separates the "SensorOffset1" from 
      the first model from the "SensorOffset1" (same name!) parameter from 
      the second model. The access would then somehow be
        parameter_object.get("SensorOffset1", first_model_key)
      vs
        parameter_object.get("SensorOffset1", second_model_key)

"""


class ModelErrorParameters:
    """
    This is essentially a dictionary that only allows setting new keys
    via self.define, not via self.__setitem__.
    """

    def __init__(self):
        self._p = {}

    @property
    def names(self):
        return list(self._p.keys())

    def define(self, name, value=None):
        self._p[name] = value

    def __getitem__(self, name):
        return self._p[name]

    def __setitem__(self, name, value):
        if name not in self.names:
            raise Exception("Call .define to define new parameters.")
        self._p[name] = value

    def update(self, names, numbers):
        assert len(names) == len(numbers)
        self._p.update(zip(names, numbers))
        return self

    def has(self, name):
        return name in self._p

    def __len__(self):
        return len(self._p)

    def __add__(self, other):
        concat = ModelErrorParameters()
        for name in self.names:
            concat.define(name, self[name])
        for name in other.names:
            concat.define(name, other[name])
        return concat

    def __str__(self):
        return str(self._p)


class LatentParameter(list):
    """
    Stores a single latent parameter as a list of tuple(key, parameter_name)
    to indicate to which (possibly multiple) individual parameters it is 
    connected. 

    Do deal with the case of vector-valued parameters, it keeps track of its 
    length `N` and its start index `start_idx` in the global parameter vector.

    Remark: The user should interact with this class only via the interfaces of
            "LatentParameters".
    """

    def __init__(self):
        self.N = None
        self.start_idx = None

    def add(self, key, name, N):
        assert not (key, name) in self
        if self.N is None:
            self.N = N
        else:
            assert self.N == N

        self.append((key, name))

    def __str__(self):
        if self.N == 1:
            idx_range = str(self.start_idx)
        else:
            idx_range = f"{self.start_idx}..{self.start_idx+self.N}"

        return f"{idx_range:10} {list.__str__(self)}"

    def extract(self, all_numbers):
        if self.N == 1:
            return all_numbers[self.start_idx]
        else:
            return all_numbers[self.start_idx : self.start_idx + self.N]


class LatentParameters(OrderedDict):
    """
    The purpose of this class is to map the named(!) parameters of multiple 
    `ModelParameters` to a vector just containing numbers and vice versa. 

    The individual `ModelErrorParameters` objects are identified by a `key`. 
    """

    def __init__(self):
        self._all_parameter_lists = {}
        self._total_length = None

    def define_parameter_list(self, parameter_list, key=None):
        assert key not in self._all_parameter_lists
        self._all_parameter_lists[key] = parameter_list

    def add(self, latent_name, parameter_name, key=None):
        assert key in self._all_parameter_lists
        assert self._all_parameter_lists[key].has(parameter_name)

        try:
            N = len(self._all_parameter_lists[key][parameter_name])
        except:
            N = 1

        if latent_name not in self:
            self[latent_name] = LatentParameter()

        self[latent_name].add(key, parameter_name, N)
        self._update_idx()

    def exists(self, parameter_name, key=None):
        for latent in self.values():
            if (key, parameter_name) in latent:
                return True
        return False

    def global_range(self, parameter_name):
        latent = self[parameter_name]
        return list(range(latent.start_idx, latent.start_idx + latent.N))

    def add_by_name(self, latent_name):
        for key, prm_list in self._all_parameter_lists.items():
            if prm_list.has(latent_name):
                self.add(latent_name, latent_name, key)

    def _update_idx(self):
        self._total_length = 0
        for key, latent in self.items():
            latent.start_idx = self._total_length
            self._total_length += latent.N or 0

    def update(self, number_vector):
        assert len(number_vector) == self._total_length
        for l in self.values():
            latent_numbers = l.extract(number_vector)
            for (key, prm_name) in l:
                self._all_parameter_lists[key][prm_name] = latent_numbers
        return self._all_parameter_lists

    def __str__(self):
        s = ""
        for key, latent in self.items():
            s += f"{key:10}: {latent} \n"
        return s

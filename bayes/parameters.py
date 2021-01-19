import copy

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


class ModelParameters:
    """
    This is essentially a dictionary that only allows setting new keys
    via self.define, not via self.__setitem__.
    """

    def __init__(self):
        self._p = {}

    @property
    def names(self):
        return list(self._p.keys())

    def define(self, name, value=0.0):
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
        concat = ModelParameters()
        for name in self.names:
            concat.define(name, self[name])
        for name in other.names:
            concat.define(name, other[name])
        return concat

    def __str__(self):
        return str(self._p)


class JointLatent:
    """
    The purpose of this class is to map the named(!) parameters of multiple 
    `ModelParameter` collections to a vector just containing numbers and 
    vice versa. 

    The individual `ModelParameter` objects are identified by a `key`. Please
    see the documentation of the individual methods for further information.
    Note that they will all refer to the following example:


    Documentation by example:

    +---------------------+                   +-----------------------+
    | model_parameters_AE |                   |  model_parameters_BE  |
    | key = AE            |                   |      key = BE         |
    |                     |  JointLatent      |                       |
    |                     |                   |                       |
    |       apple +------------> 0            |    + football         |
    |                     |                   |    |                  |
    |    soccer +--------------> 1 <---------------+            egg   |
    |                     |                   |                       |
    |            fries +  |      2 <--------------+ holiday           |
    |                  |  |                   |                       |
    |   vacation       +-------> 3 <---------------+ chips            |
    |                     |                   |                       |
    |           blue +---------> 4            |                       |
    |                     |                   |                       |
    +---------------------+                   +-----------------------+


    """

    def __init__(self):
        self.all_parameters = {}
        self._index_mapping = []
        self._parameter_mapping = {}

    def add_model_parameters(self, model_parameters, key=None):
        """
        Adds a `ModelParameter`.

        .add_model_parameters(model_parameters_AE, AE)
        .add_model_parameters(model_parameters_BE, BE)
        """
        assert key not in self.all_parameters
        self.all_parameters[key] = model_parameters

    def _add(self, name, key, index=None):
        """
        Internal method to add parameter `name` of `key` to the `index`th latent
        parameter.
        """
        assert not self.exists(name, key)
        assert name in self.all_parameters[key].names

        if index is None:
            index = len(self._index_mapping)
            self._index_mapping.append({})

        self._index_mapping[index][key] = name
        self._parameter_mapping[(name, key)] = index
        return index

    def exists(self, name, key):
        """
        .exists(soccer, AE) == True
        .exists(soccer, BE) == False
        """
        return (name, key) in self._parameter_mapping

    def add(self, name, key=None):
        """
        Add parameter `name` of `key` as a _new_ latent variable. 

        .add(apple, AE) == 0
        .add(soccer, AE) == 1
        .add(holiday, BE) == 2
        .add(fries, AE) == 3
        .add(blue, AE) == 4
        """
        return self._add(name, key)

    def add_shared(self, index, name, key=None):
        """
        Add parameter `name` of `key` to an _existing_ `index`th latent variable.

        .add_shared(1, football, BE) == 1
        .add_shared(3, chips, BE) == 3
        """
        return self._add(name, key, index)

    def add_by_name(self, name):
        """
        Sets all the parameters `name` to the same latent variable.
        """
        entry = None
        for key, model_parameters in self.all_parameters.items():
            if model_parameters.has(name):
                if entry is None:
                    entry = self.add(name, key)
                else:
                    self.add_shared(entry, name, key)
        return entry

    def parameter(self, index):
        """
        Returns a `key`:`name` dictionary of the `index`th latent variable.

        .parameter(0) == {AE:apple}
        .parameter(1) == {AE:soccer, BE:football}
        """
        return self._index_mapping[index]

    def indices_of(self, key):
        """
        Returns the indices of all latent parameters of `key`.

        .indices_of(AE) == [0, 1, 3, 4]
        .indices_of(BE) == [1, 2, 3]
        """
        return [i for i, mapping in enumerate(self._index_mapping) if key in mapping]

    def index_of(self, name, key=None):
        """
        Returns the index of parameter `name` of `key`.

        .index_of(apple, AE) == 0
        .index_of(chips, BE) == 3
        """
        return self._parameter_mapping[(name, key)]

    def __len__(self):
        """
        Returns the number of latent variables.

        .__len__() == 5
        """
        return len(self._index_mapping)

    def __getitem__(self, index):
        """
        Pythonic wrapper for self.parameter.
        """
        return self.parameter(index)

    def update(self, numbers, return_copy=True):
        """
        Returns a dictionary key:ModelParameter where the latent variables
        are set to `numbers`.

        .update(self, 0, 10, 20, 30, 40) ==
         {AE: [salmon[unchanged], apple[0], soccer[10], fries[30], 
               vacation[unchanged], blue[40],
          BE: [football[10], egg[unchanged], holiday[20], chips[30]]}
        """
        assert len(numbers) == len(self)

        if return_copy:
            updated_parameters = copy.deepcopy(self.all_parameters)
        else:
            updated_parameters = self.all_parameters

        for number, parameters in zip(numbers, self):
            for (key, name) in parameters.items():
                updated_parameters[key][name] = number
        return updated_parameters

    def __str__(self):
        return "\n".join([f"{i:} {prm}" for i, prm in enumerate(self)])


class UncorrelatedNormalPrior:
    def __init__(self, latent):
        self.latent = latent
        if len(self.latent) != 0:
            raise RuntimeError(
                "This class takes now takes care of setting the"
                "latent parameters. You may not define them beforehand!"
            )
        self.distributions = []

    def add(self, name, mean, sd, key=None):
        entry = self.latent.add(name, key)
        self.distributions.append((mean, sd))
        return entry

    def add_shared(self, key_name_pairs, mean, sd):
        index = None 
        for (key, name) in key_name_pairs:
            if index is None:
                entry = self.add(name, mean, sd, key)
            else:
                self.latent.add_shared(index, name, key)
        return index

    def to_MVN(self):
        from bayes.vb import MVN
        import numpy as np

        N = len(self.distributions)

        mean = [d[0] for d in self.distributions]
        prec = [1 / d[1] ** 2 for d in self.distributions]

        return MVN(mean, np.diag(prec))

    def __str__(self):
        return str(self.prm)

    def __len__(self):
        return len(self.distributions)

    def distribution_of(self, name, key):
        return self.distributions[self.latent.index_of(name, key)]

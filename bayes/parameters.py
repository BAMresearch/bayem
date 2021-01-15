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
    def __init__(self):
        self.all_parameters = {}
        self._index_mapping = []
        self._parameter_mapping = {}

    def add_model_parameters(self, model_parameters, key):
        assert key not in self.all_parameters
        self.all_parameters[key] = model_parameters

    def _add(self, name, key, index=None):

        assert not self.exists(name, key)

        if index is None:
            index = len(self._index_mapping)
            self._index_mapping.append({})

        self._index_mapping[index][key] = name
        self._parameter_mapping[(name, key)] = index
        return index

    def exists(self, name, key):
        return (name, key) in self._parameter_mapping

    def add_shared(self, index, name, key):
        return self._add(name, key, index)

    def add(self, name, key):
        return self._add(name, key)

    def parameter(self, index):
        return self._index_mapping[index]

    def indices_of(self, key):
        return [i for i, mapping in enumerate(self._index_mapping) if key in mapping]

    def index_of(self, name, key):
        return self._parameter_mapping[(name, key)]

    def __len__(self):
        return len(self._index_mapping)

    def __getitem__(self, index):
        return self.parameter(index)

    def update(self, numbers):
        assert len(numbers) == len(self)
        for number, parameters in zip(numbers, self):
            for (key, name) in parameters.items():
                self.all_parameters[key][name] = number
        return self.all_parameters

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

    def add_shared(self, index, name, key):
        return self.latent.add_shared(index, name, key)

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

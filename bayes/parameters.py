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
        self.p = {}

    def define(self, name, value=None):
        self.p[name] = value

    def __getitem__(self, name):
        return self.p[name]

    def __contains__(self, name):
        return name in self.p

    def __setitem__(self, name, value):
        if name not in self:
            raise Exception("Call .define to define new parameters.")
        self.define(name, value)

    def __add__(self, other):
        concat = ModelErrorParameters()
        for name, value in self.p.items():
            concat.define(name, value)
        for name, value in other.p.items():
            concat.define(name, value)
        return concat

    @property
    def names(self):
        return list(self.p.keys())

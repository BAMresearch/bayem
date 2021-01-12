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
        self.p = {}

    @property
    def names(self):
        return list(self.p.keys())

    def define(self, name, value=0.0):
        self.p[name] = value

    def __getitem__(self, name):
        return self.p[name]

    def __setitem__(self, name, value):
        if name not in self.names:
            raise Exception("Call .define to define new parameters.")
        self.p[name] = value

    def update(self, names, numbers):
        assert len(names) == len(numbers)
        self.p.update(zip(names, numbers))
        return self

    def has(self, name):
        return name in self.p


class JointParameterList:
    def __init__(self, model_parameters, shared=None):
        """
        This class allows you to combine multiple individual parameter lists
        of different models to a joint one. 

        Features:
        ---------

        * shared parameters:

            TODO
        
        * latent parameters:

            Calling "update" method with a vector of numbers provided by the 
            inference schemes, it will perform the mapping to the named
            model_parameters and update their numbers.

        Single model parameter list
        ---------------------------

        model_parameters:
            `ModelParameters` object

        You simply ignore all the "shared" and "key" arguments of this class
        and just define the prior distribution according to the parameter names
        of `model_parameters`.
        In fact, "shared" and "key" _has_ to be kept at "None"!

        
        Multiple parameter lists
        ------------------------

        model_parameters:
            dictionary of type [key --> ModelParameters]

        shared:
            List of parameter names that get the same prior distribution.

        In this more elaborate use of the class, you have to define and refer 
        to parameters by their "name" _and_ their "key". This key separates
        potentially identical (=same names) parameter lists. Example:
            model_parameters:
                "ModelA" : [P1, P2, P3, P4]
                "ModelB" : [P2, P3, P9]
            shared:
                [P3]

        That would mean that you can provide a common prior distribution for
        both ModelA.P3 and ModelB.P3. They both get one entry in the final 
        parameter vector and changing this entry will modify both ModelA.P3
        and ModelB.P3. 
        The parameter P2 also present in both models, but not shared. Here,
        you have to specify the key ("ModelA" or "ModelB") in the prior 
        definition and it will result in two entries in the final parameter
        vector.


        Notes
        -----

        If you pass a "ModelParameters" instance to this class, it will be 
        modified in ".update". This may cause confusion and we can actually 
        think about immutable "ModelParameters", where the ".update" keeps
        the original self.model_parameters and returns a modified copy.
        """
        self.ps = model_parameters
        self.shared = shared or []  # avoid lists in default args!

        self.latent_parameters = []

        # deal with special (actually simpler) case of a single parameter list
        if type(model_parameters) is ModelParameters:
            assert shared is None
            self.ps = {"default": model_parameters}
            self.shared = model_parameters.names

    def set_latent(self, name, key=None):
        """
        Sets a specific parameter latent such that it is updated within .update

        name:
            parameter name that must match one in global parameters
        
        key:
            The key indicates to which the parameter group the "name" belongs
            to. If key is None, this assumes a _shared_ parameter.
        """
        latent = []

        if key is not None:
            assert name in self.ps[key].names
            assert name not in self.shared
            latent.append((key, name))

        else:
            assert name in self.shared
            for key, model_parameters in self.ps.items():
                if name in model_parameters.names:
                    latent.append((key, name))

        if latent in self.latent_parameters:
            index = self.latent_parameters.index(latent)
            raise RuntimeError(
                f"Parameter {latent} is already set as latent"
                " in position {index}! Must only be added once!"
            )
        self.latent_parameters.append(latent)

    def update(self, number_vector):
        """
        Updates all ModelParameters known to the prior distribution according
        to the latent parameters.
        """
        assert len(number_vector) == len(self.latent_parameters)
        for number, latents in zip(number_vector, self.latent_parameters):
            for latent in latents:
                key, name = latent
                self.ps[key][name] = number

    def __str__(self):
        s = ""
        for i, latents in enumerate(self.latent_parameters):
            for latent in latents:
                s += f"{i:3d} {latent}\n"
        return s


class UncorrelatedNormalPrior:
    def __init__(self, parameter_list):
        self.prm = parameter_list
        if self.prm.latent_parameters:
            raise RuntimeError(
                "This class takes now takes care of setting the"
                "latent parameters. You may not define them beforehand!"
            )
        self.distributions = []

    def add(self, name, mean, sd, key=None):
        self.prm.set_latent(name, key)
        self.distributions.append((mean, sd))

    def to_MVN(self):
        from bayes.vb import MVN
        import numpy as np

        N = len(self.distributions)

        mean = [d[0] for d in self.distributions]
        prec = [1 / d[1] ** 2 for d in self.distributions]

        return MVN(mean, np.diag(prec))

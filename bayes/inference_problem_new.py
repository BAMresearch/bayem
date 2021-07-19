# standard library imports
from copy import deepcopy as dc

# local imports
from bayes.priors import LogPriorNormal
from bayes.priors import LogPriorLognormal
from bayes.priors import LogPriorUniform
from bayes.priors import LogPriorWeibull
from bayes.subroutines import underlined_string, tcs


class InferenceProblem:

    def __init__(self, name):
        """
        Parameters
        ----------
        name : string
            This is the name of the problem and has only descriptive value, for
            example when working with several inference problems
        """

        # the name of the problem
        self.name = name

        # this is the central parameter dictionary of the problem; it contains
        # all defined parameters ('const' and 'calibration' ones); the keys of
        # this dictionary are the parameter names; note that each parameter must
        # have a unique name in the problem; the values of this dictionary are
        # again dictionaries with the following key-value pairs:
        # 'index': int or None (the index in the theta-vector (see self.loglike)
        #          for 'calibration'-parameter; None for 'const'-parameters)
        # 'type':  string (either 'model', 'prior' or 'noise' depending on where
        #          the parameter appears)
        # 'role':  string (either 'const' for a constant parameter or
        #          'calibration' for a calibration parameter)
        # 'prior': object or None (the prior-object of the 'calibration'-
        #          parameter; None for 'const'-parameters)
        # 'value': float or None (defines the value for 'const'-parameters;
        #          None for 'calibration'-parameters)
        # 'info':  string (a short explanation of the parameter)
        self._prm_dict = {}

        # the number of currently defined 'calibration'-parameters
        self.n_calibration_prms = 0

        # this dictionary is intended for storing the measured data from
        # experiments (see self.add_experiment); this dict is managed
        # internally and should not be edited directly
        self._experiments = {}

        # here, the available prior classes are stored; additional ones can be
        # added via self.add_prior_class; see also priors.py; this dict is
        # managed internally and should not be edited directly
        self._logprior_classes = {'normal': LogPriorNormal,
                                  'lognormal': LogPriorLognormal,
                                  'uniform': LogPriorUniform,
                                  'weibull': LogPriorWeibull}

        # dictionary of the problem's priors; the items will have the structure
        # <prior name> : <prior object>; this dict is managed internally and
        # should not be edited directly
        self._priors = {}

        # the forward model of the problem will be written to this attribute;
        # it is managed internally and should not be edited directly
        self._forward_model = None

        # a dictionary for the problem's noise models; note that noise models
        # are defined sensor-specific, so the items of this dict are of the
        # structure <sensor name> : <noise model object>; this dict is managed
        # internally and should not be edited directly
        self._noise_models = {}

        # these sets are for collecting all names of the problem's input and
        # output sensors; note that self._input_sensors has no use yet except
        # for debugging purposes; these sets are managed internally and should
        # not be edited directly
        self._input_sensors = set()
        self._output_sensors = set()

    def __str__(self):
        """
        Allows to print relevant problem information by calling print(problem)
        if problem is an instance of InferenceProblem.
        """

        # contains the name of the inference problem
        title_string = underlined_string(self.name, n_empty_start=2)

        # provide a parameter overview sorted by their roles and types
        n_prms = len(self._prm_dict.keys())
        prms_string = underlined_string("Parameter overview", symbol="-")
        prms_string += f"Number of parameters:   {n_prms}\n"
        prms_roles_types = {'model': [], 'prior': [], 'noise': [],
                            'calibration': [], 'const': []}
        for prm_name, prm_dict in self._prm_dict.items():
            prms_roles_types[prm_dict['role']].append(prm_name)
            prms_roles_types[prm_dict['type']].append(prm_name)
        for group, prms in prms_roles_types.items():
            prms_string += tcs(f'{group.capitalize()} parameters', prms)

        # provide an overview over the 'const'-parameter's values
        const_prms_str = underlined_string("Constant parameters", symbol="-")
        w = len(max(prms_roles_types['const'], key=len)) + 2
        for prm_name in prms_roles_types['const']:
            prm_value = self._prm_dict[prm_name]['value']
            const_prms_str += tcs(prm_name, f"{prm_value:.2f}", col_width=w)

        # additional information on the problem's parameters
        prms_info_str = underlined_string("Parameter explanations", symbol="-")
        w = len(max(self._prm_dict.keys(), key=len)) + 2
        for prm_name, prm_dict in self._prm_dict.items():
            prms_info_str += tcs(prm_name, f"{prm_dict['info']}", col_width=w)

        # include information on the defined priors
        prior_str = underlined_string("Priors defined", symbol="-")
        w = len(max(self._priors.keys(), key=len)) + 2
        for prior_name, prior_obj in self._priors.items():
            prior_str += tcs(prior_name, str(prior_obj), col_width=w)

        # concatenate the string and return it
        full_string = title_string + prms_string + const_prms_str
        full_string += prms_info_str + prior_str
        return full_string

    def add_parameter(self, prm_name, prm_type, const=None, prior=None,
                      info="No explanation provided"):
        """
        Adds a parameter to the inference problem.

        Parameters
        ----------
        prm_name : string
            The name of the parameter which should be added to the problem
        prm_type : string
            Either 'model' (for a model parameter), 'prior' (for a prior
            parameter) or 'noise' (for a noise parameter); note that 'prior'
            parameter do not have to be added by hand; they are added
            internally when 'calibration'-parameters are added
        const : float or None, optional
            If the added parameter is a 'const'-parameter, the corresponding
            value has to be specified by this argument
        prior : tuple of two elements or None, optional
            If the added parameter is a 'calibration'-parameter, this argument
            has to be given as a 2-tuple. The first element (a string) defines
            the prior-type (must be a key of self._logprior_classes); the second
            element must be a dictionary stating the prior's parameters; its
            definition is identical to the one of prm_dict explained in the
            docstring of the self._add_prior method.
        info : string, optional
            Short explanation on the added parameter
        """

        # make sure the given prm_type is valid
        if prm_type not in ['model', 'prior', 'noise']:
            raise RuntimeError(
                f"Unknown parameter type: prm_type = {prm_type} \n" +
                f"Valid arguments are 'model', 'prior' or 'noise'."
            )

        # exactly one of the const and prior key word arguments must be given
        if const is not None and prior is not None:
            raise RuntimeError(
                f"You must specify either the 'const' or the 'prior' key " +
                f"argument. You have specified both."
            )
        if const is None and prior is None:
            raise RuntimeError(
                f"You must specify either the 'const' or the 'prior' key " +
                f"argument. You have specified none."
            )

        # check whether the parameter name was used before; note that all
        # parameters (across types!) must have unique names
        if prm_name in self._prm_dict.keys():
            raise RuntimeError(
                f"A parameter with name '{prm_name}' has already been" +
                f" defined. Please choose another name."
            )

        # add the parameter to the central parameter dictionary
        prm_role = 'calibration' if const is None else 'const'
        if const is None:
            # in this case we are adding a 'calibration'-parameter
            prm_index = self.n_calibration_prms
            self.n_calibration_prms += 1
            prm_value = None
            prior_type = prior[0]  # e.g. 'normal', 'lognormal', etc.
            prior_dict = prior[1]  # dictionary with parameter-value pairs
            prior_parameter_names = []
            for prior_parameter_name, value in prior_dict.items():
                # create unique name for this prior parameter
                new_name = f"{prior_parameter_name}_{prm_name}"
                prior_parameter_names.append(new_name)
                # the prior parameter is considered a constant parameter
                default_info = f"{prior_type.capitalize()} prior's parameter "
                default_info += f"for calibration-parameter '{prm_name}'"
                self.add_parameter(new_name, 'prior', const=value,
                                   info=default_info)
            prior_name = f"{prm_name}_{prior_type}"  # unique name of this prior
            prm_prior = self._add_prior(prior_name, prior_type, prm_name,
                                        prior_parameter_names)
        else:
            # in this case we are adding a 'const'-parameter
            prm_index = None
            prm_prior = None
            prm_value = const
        self._prm_dict[prm_name] = {'index': prm_index,
                                    'type': prm_type,
                                    'role': prm_role,
                                    'prior': prm_prior,
                                    'value': prm_value,
                                    'info': info}

    def remove_parameter(self, prm_name):
        """
        Removes a parameter from the inference problem.

        Parameters
        ----------
        prm_name : string
            The name of the parameter to be removed
        """
        # check if the given parameter exists
        if prm_name not in self._prm_dict.keys():
            raise RuntimeError(
                f"A parameter with name '{prm_name}' has not been defined yet."
            )
        # different steps must be taken depending on whether the parameter which
        # should be removed is a 'const'- or a 'calibration'-parameter
        if self._prm_dict[prm_name]['index'] is None:
            # in this case prm_name refers to a constant parameter; hence, we
            # can simply remove this parameter without side effects
            del self._prm_dict[prm_name]
        else:
            # in this case prm_name refers to a calibration parameter; hence we
            # need to remove the prior-parameter and the prior-object; also, we
            # have to correct the index values of the remaining calibration prms
            for prior_prm in self._prm_dict[prm_name]['prior'].prms:
                self.remove_parameter(prior_prm)
            del self._priors[self._prm_dict[prm_name]['prior'].name]
            del self._prm_dict[prm_name]['prior']
            del self._prm_dict[prm_name]
            # correct the indices of the remaining 'calibration'-parameters
            idx = 0
            for name, prm_dict in self._prm_dict.items():
                if prm_dict['index'] is not None:
                    prm_dict['index'] = idx
                    idx += 1
            self.n_calibration_prms -= 1

    def change_parameter_role(self, prm_name, const=None, prior=None):
        """
        Performs the necessary tasks to change a parameter's role in the problem
        definition. A parameter's role can either be changes from 'const' to
        'calibration' or from 'calibration' to 'const'.

        Parameters
        ----------
        prm_name : string
            The name of the parameter whose role should be changed
        const : float or None, optional
            If the new role is 'const', the corresponding value has to be
            specified by this argument
        prior : tuple of two elements or None, optional
            If a 'const'-parameter should be changed to a 'calibration'
            parameter, this argument has to be given as a 2-tuple. The first
            element (a string) defines the prior-type (must be a key of
            self._logprior_classes); the second element must be a dictionary
            stating the prior parameters; its definition is identical to the
            one of prm_dict in the self._add_prior method.
        """
        # check if the given parameter exists
        if prm_name not in self._prm_dict.keys():
            raise RuntimeError(
                f"A parameter with name '{prm_name}' has not been defined yet."
            )
        # exactly one of the const and prior key word arguments must be given
        if const is not None and prior is not None:
            raise RuntimeError(
                f"You must specify either the 'const' or the 'prior' key " +
                f"argument. You have specified both."
            )
        if const is None and prior is None:
            raise RuntimeError(
                f"You must specify either the 'const' or the 'prior' key " +
                f"argument. You have specified none."
            )
        # change the parameter's role by first removing it from the problem, and
        # then adding it again in its new role
        prm_type = self._prm_dict[prm_name]['type']
        prm_info = self._prm_dict[prm_name]['info']
        self.remove_parameter(prm_name)
        self.add_parameter(prm_name, prm_type, const=const, prior=prior,
                           info=prm_info)

    def change_parameter_info(self, prm_name, new_info):
        """
        Changes the info-string of a given parameter

        Parameters
        ----------
        prm_name : string
            The name of the parameter whose info-string should be changed
        new_info : string
            The new string for the explanation of parameter prm_name
        """
        # check if the given parameter exists
        if prm_name not in self._prm_dict.keys():
            raise RuntimeError(
                f"A parameter with name '{prm_name}' has not been defined yet."
            )
        # change the info-string
        self._prm_dict[prm_name]['info'] = new_info

    def change_constant(self, prm_name, new_value):
        """
        Changes the value of a 'const'-parameter, i.e. a constant parameter of
        the inference problem.

        Parameters
        ----------
        prm_name : string
            The name of the 'const'-parameter whose value should be changed
        new_value : float
            The new value that prm_name should assume
        """
        # check if the given parameter exists
        if prm_name not in self._prm_dict.keys():
            raise RuntimeError(
                f"A parameter with name '{prm_name}' has not been defined yet."
            )
        # check if the given parameter is a constant
        if self._prm_dict[prm_name]['role'] != "const":
            raise RuntimeError(
                f"The parameter '{prm_name}' is not a constant!"
            )
        # change the parameter's value
        self._prm_dict[prm_name]['value'] = new_value

    def _add_prior(self, name, prior_type, prm_dict, ref_prm):
        """
        Adds a prior-object of a calibration parameter to the internal prior
        dictionary.

        Parameters
        ----------
        name : string
            Unique name of the prior; usually this name has the structure
            <ref_prm>_<prior_type>
        prior_type : string
            Defines the prior type; must be one of the keys in the internal
            dictionary self._logprior_classes.
        prm_dict : dict
            States the prior's parameters as 'name : value' items; the order
            of these items is important (!); they must correspond to the order
            of how the prior's __call__ method interprets its 'prms' argument;
            check out the classes in priors.py for a better understanding
        ref_prm : string
            The name of the problem's calibration parameter the prior refers to
            (a prior is always defined for a specific calibration parameter)

        Returns
        -------
        object
            The prior-object which is also written to the internal prior
            dictionary self._priors
        """
        logprior_class = self._logprior_classes[prior_type]
        self._priors[name] = dc(logprior_class(prm_dict, ref_prm, name))
        return self._priors[name]

    def add_prior_class(self, name, prior_class):
        """
        This method allows to add a user-defined prior-class to the problem, so
        it can be used in the parameter definitions.

        Parameters
        ----------
        name : string
            Name of the prior class to be added
        prior_class : class
            The user-defined prior-class; see priors.py for examples.
        """
        # add the class to the internal prior-class dictionary if the given
        # name has not been used for one of the existing classes
        if name in self._logprior_classes.keys():
            raise RuntimeError(
                f"A prior class with name '{name}' already exists. " +
                f"Please choose another name."
            )
        self._logprior_classes[name] = prior_class

    def check_problem_consistency(self):
        """
        Conducts various checks to make sure the problem definition does not
        contain any inconsistencies.
        """

        # check if the central components have been added to the problem:
        # parameters, priors, a forward model, experiments and noise models;
        # the following statements assert that the corresponding attributes are
        # not empty or None
        assert self._prm_dict, "No parameters have been defined yet!"
        assert self._priors, "Found no priors in the problem definition!"
        assert self._forward_model, "No forward model has been defined yet!"
        assert self._experiments, "No experiments have been defined yet!"
        assert self._noise_models, "No noise models have been defined yet!"

        # check if all constant parameters have values assigned
        for prm_dict in self._prm_dict.values():
            if prm_dict['role'] == 'const':
                assert prm_dict['value'] is not None

        # check if all parameters of the forward model appear in self._prm_dict
        # and if they have the correct type
        for model_prm in self._forward_model.prms_def:
            assert model_prm in self._prm_dict.keys()
            assert self._prm_dict[model_prm]['type'] == "model"

        # check if all parameters of the noise model appear in self._prm_dict
        # and if they have the correct type
        for noise_model in self._noise_models.values():
            for noise_prm in noise_model.prms_def:
                assert noise_prm in self._prm_dict.keys()
                assert self._prm_dict[noise_prm]['type'] == "noise"

        # check if all prior objects in self._priors are consistent in terms of
        # their parameters; each one of them must appear in self._prm_dict
        assert len(self._priors) == self.n_calibration_prms
        for prior_obj in self._priors.values():
            prior_prms = prior_obj.prms
            for prior_prm in prior_prms:
                assert prior_prm in self._prm_dict.keys()
                assert self._prm_dict[prior_prm]['type'] == 'prior'
                assert self._prm_dict[prior_prm]['role'] == 'const'

        # check if the prior-parameters of each calibration parameter exist in
        # the problem's parameter dictionary
        for prm_name, prm_dict in self._prm_dict.items():
            if prm_dict['role'] == 'calibration':
                prior_prms = prm_dict['prior'].prms
                for prior_prm in prior_prms:
                    assert prior_prm in self._prm_dict.keys()
                    assert self._prm_dict[prior_prm]['role'] == "const"

        # check the indices of the calibration parameters; the order of how
        # these parameters appear in self._prm_dict matter! the first
        # calibration parameter must have index 0, while the following one must
        # have index 1 and so forth; the total number of calibration indices
        # must be identical to self.n_calibration_prms
        idx_list = []
        for prm_name, prm_dict in self._prm_dict.items():
            if prm_dict['role'] == 'calibration':
                idx_list.append(prm_dict['index'])
        assert len(idx_list) == self.n_calibration_prms
        assert idx_list[0] == 0  # first index must be zero
        if self.n_calibration_prms > 1:
            diff_list = [x - idx_list[i - 1] for i, x in enumerate(idx_list)][
                        1:]
            assert max(diff_list) == 1  # no index can be skipped

        # check if all output sensors defined in the experiments have been
        # assigned to a noise model
        assert set(self._noise_models.keys()) == self._output_sensors

    def add_experiment(self, exp_name, exp_input, exp_output):
        """
        Adds a single experiment to the inference problem. Here, an experiment
        is defined by having an input sensor and an output sensor, each with a
        specific value corresponding to the conducted experiment.

        Parameters
        ----------
        exp_name : string
            The name of the experiment, e.g. "Exp_20May.12"; if an experiment
            with a similar name has already been added, it will be overwritten.
        exp_input : tuple with two elements
            The 1st element is a string specifying the input sensor, while the
            2nd element is the corresponding value, e.g. ("ForceSensor", 1.2)
        exp_output : tuple with two elements
            The 1st element is a string specifying the output sensor, while the
            2nd element is the corresponding value, e.g. ("LengthSensor", 0.02)
        """

        # throw warning when the experiment name was defined before
        if exp_name in self._experiments.keys():
            print(f"WARNING - Experiment '{exp_name}' is already defined" +
                  f" and will be overwritten!")

        # add the experiment to the central dictionary
        self._experiments[exp_name] = {'input': {'sensor': exp_input[0],
                                                 'value': exp_input[1]},
                                       'output': {'sensor': exp_output[0],
                                                  'value': exp_output[1]}}

        # bookkeeping of the used sensors; note that these are sets, not lists
        self._input_sensors.add(exp_input[0])
        self._output_sensors.add(exp_output[0])

    def get_parameters(self, theta, prm_names_):
        """
        Extracts the numeric values for a given list of parameters from the
        parameter vector and the constant parameters of the problem.

        Parameters
        ----------
        theta : array_like
            A numeric parameter vector passed to the loglike and logprior
            method. Which parameters these numbers refer to can be checked
            by calling self.theta_explanation() once the problem is set up.
        prm_names_ : string or list of strings
            The names of the parameters whose values should be returned. The
            returned values will be in the same order as given in prm_names_.

        Returns
        -------
        prms : list
            Numeric values for the parameters stated in prm_names_
        """
        # if prm_names_ is given as a single string, it is converted to a list
        prm_names = prm_names_ if type(prm_names_) is list else [prm_names_]
        prms = []
        for prm_name in prm_names:
            idx = self._prm_dict[prm_name]['index']
            if idx is None:
                # in this case, the parameter is a constant and hence not read
                # from theta
                prms.append(self._prm_dict[prm_name]['value'])
            else:
                # in this case, the parameter is a calibration parameter, and
                # its value is read from theta
                prms.append(theta[idx])
        return prms

    def theta_explanation(self):
        """
        Prints out how the theta-vector, which is the numeric parameter vector
        that is given to the self.loglike and self.logprior methods, is
        interpreted with respect to the problems parameters. The printout will
        tell you which parameter is connected to which index of theta.
        """

        # an explanation is not printed if the problem is inconsistent
        self.check_problem_consistency()

        # assemble the parameter's names in the order as they appear in theta
        # from the index information in self._prm_dict, and print the result
        theta_names = []
        for prm_name, prm_dict in self._prm_dict.items():
            if prm_dict['index'] is not None:
                theta_names.append(prm_name)
        print("\n---------------------")
        print("| Theta | Parameter |")
        print("| index |   name    |")
        print("|-------------------|")
        for i, prm_name in enumerate(theta_names):
            print(f"|{i:5d} --> {prm_name:<9s}|")
        print("---------------------\n")

    def add_forward_model(self, forward_model_class, prms_def):
        """
        Adds the given forward model to the inference problem.

        Parameters
        ----------
        forward_model_class : class
            The class defining the forward model; check out forward_model.py to
            see a template for the forward model definition.
        prms_def : list
            The names (strings) of the forward model's parameters; note that
            these parameters need to be added to the problem before adding the
            forward model. The order reflected in prms_def defines how numeric
            values for these parameters will be given to the '__call__' method
            of the forward model.
        """

        # check if all given model parameters have already been added to the
        # inference problem
        for prm_name in prms_def:
            if prm_name not in self._prm_dict.keys():
                raise RuntimeError(
                    f"The model parameter '{prm_name}' has not been defined " +
                    f"yet.\nYou have to add all model parameters to the " +
                    f"problem before adding the model.\nYou can use the " +
                    f"'add_parameter' method for this purpose."
                )

        # instantiate an object of the given class and define it as the
        # inference problem's forward model
        self._forward_model = forward_model_class(prms_def)

    def evaluate_model_error(self, theta, experiments=None, key="sensor"):
        """
        Evaluates the model error for the given parameter vector and the given
        experiments.

        Parameters
        ----------
        theta : array_like
            A numeric vector for which the model error should be evaluated.
            Which parameters these numbers refer to can be checked by calling
            self.theta_explanation() once the problem is set up.
        experiments : dict or None, optional
            Contains all or some of the experiments added to the inference
            problem. Check out the self.add_experiment method to see how this
            dictionary must be structured. If this argument is None then all
            experiments defined in the problem (self.experiments) are used.
        key : string, optional
            Either 'sensor' or 'experiment' defining if the returned model error
            dictionary will have the problem's sensors or experiments as keys

        Returns
        -------
        model_error : dict
            Contains either the problem's sensors or experiments as keys, and
            states the corresponding model errors as values.
        """

        # check the input
        if key not in ["sensor", "experiment"]:
            raise RuntimeError(
                f"This method requires key='sensor' or key='experiment' as " +
                f"argument. You defined key='{key}'"
            )

        # if experiments is not further specified all experiments added to the
        # problem will be accounted for when computing the model error
        if experiments is None:
            experiments = self._experiments

        # the model error is computed within the model object
        prms_model = self.get_parameters(theta, self._forward_model.prms_def)
        model_error = self._forward_model.error(prms_model, experiments,
                                                key=key)

        return model_error

    def add_noise_model(self, sensor, noise_model_class, prms_def):
        """
        Adds a sensor-specific noise model to the inference problem.

        Parameters
        ----------
        sensor : string
            The sensor type the noise model refers to, e.g. "ForceSensor". Note
            that this sensor type must appear as output sensor in the problem's
            experiments for the noise model to be used.
        noise_model_class : class
            The noise model's class, e.g. NormalNoise; check out noise.py to
            see some noise model classes
        prms_def : list
            The names (strings) of the noise model's parameters; note that these
            parameters need to be added to the problem before adding the noise
            model. The order reflected in prms_def defines how numeric values
            for these parameters will be given to the 'loglike_contribution'
            method of the noise model.
        """

        # check if all given noise model parameters have already been added to
        # the inference problem
        for prm_name in prms_def:
            if prm_name not in self._prm_dict.keys():
                raise RuntimeError(
                    f"The noise model parameter '{prm_name}' has not been " +
                    f"defined yet.\nYou have to add all noise model " +
                    f"parameters to the problem before adding the noise " +
                    f"model.\nYou can use the 'add_parameter' method for " +
                    f"this purpose."
                )

        # instantiate an object of the given class and define it the
        # inference problem's noise model for the given sensor type
        self._noise_models[sensor] = noise_model_class(prms_def)

    def logprior(self, theta):
        """
        Evaluates the log-prior function of the problem at theta.

        Parameters
        ----------
        theta : array_like
            A numeric vector for which the log-likelihood function should be
            evaluated. Which parameters these numbers refer to can be checked
            by calling self.theta_explanation() once the problem is set up.

        Returns
        -------
        lp : float
            The evaluated log-prior function for the given theta-vector.
        """
        lp = 0.0
        for prior in self._priors.values():
            prms = self.get_parameters(theta, prior.prms_def)
            lp += prior(prms)
        return lp

    def loglike(self, theta):
        """
        Evaluates the log-likelihood function of the problem at theta.

        Parameters
        ----------
        theta : array_like
            A numeric vector for which the log-likelihood function should be
            evaluated. Which parameters these numbers refer to can be checked
            by calling self.theta_explanation() once the problem is set up.

        Returns
        -------
        ll : float
            The evaluated log-likelihood function for the given theta-vector.
        """

        # the model error is returned for each sensor because the noise models
        # are defined for each sensor individually
        model_error_dict = self.evaluate_model_error(theta, key="sensor")

        # compute the contribution to the log-likelihood function for the
        # model error of each sensor type, and sum it all up
        ll = 0.0
        for sensor, me_vector in model_error_dict.items():
            noise_model = self._noise_models[sensor]  # the sensor's noise model
            prms_noise = self.get_parameters(theta, noise_model.prms_def)
            ll += noise_model.loglike_contribution(me_vector, prms_noise)

        return ll

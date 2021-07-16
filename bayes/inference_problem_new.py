# standard library imports
from copy import deepcopy as dc

# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports
from priors import LogPriorNormal
from priors import LogPriorLognormal
from priors import LogPriorUniform
from priors import LogPriorWeibull

def underlined_string(string, symbol="=", n_empty_start=1, n_empty_end=1):
    n_chars = len(string)
    underline_string = n_chars * symbol
    empty_lines_start = n_empty_start * "\n"
    empty_lines_end = n_empty_end * "\n"
    result_string = string + "\n" + underline_string
    result_string = empty_lines_start + result_string + empty_lines_end
    return result_string

def sub_when_empty(string, empty_str="-"):
    if len(string) > 0:
        result_string = string
    else:
        result_string = empty_str
    return result_string

def tcs(string_1, string_2, sep=":", col_width=24, empty_str="-", par=True):
    # two-column-string
    result_string = f"{string_1+sep:{col_width}s}{sub_when_empty(string_2, empty_str=empty_str)}"
    if par:
        result_string += "\n"
    return result_string

def mcs(string_list, col_width=18, par=False):
    # multi-column-string
    result_string = ""
    for string in string_list:
        result_string += f"{string:{col_width}}"
    if par:
        result_string += "\n"
    return result_string


class InferenceProblem:

    def __init__(self, name):

        # this is the name of the problem; this attribute has only descriptive
        # value, for example when working with several inference problems
        self.name = name

        # this attributes is a list of all defined parameters in the problem;
        # it is managed internally and should not be edited directly
        self._prm_names = []

        # here the parameter names are associated with their respective type
        # from a perspective of the problem setup; a 'model'-parameter is a
        # parameter of the forward model; a 'prior'-parameter is a parameter
        # required to describe a prior distribution and a 'noise'-parameter is a
        # parameter which describes some noise model in the problem definition;
        # the 'calibration'-parameters are supposed to be calibrated while the
        # 'const'-parameters have a fixed value and are not calibrated; note
        # that all parameter names must be unique over all five groups; also
        # note that the union of the 'model', 'prior' and 'noise' parameters is
        # identical to the union of the 'calibration' and 'const' parameters;
        # this dict is managed internally and should not be edited directly
        self._prm_names_dict = {'model': [],
                                'prior': [],
                                'noise': [],
                                'calibration': [],
                                'const': []}

        # this is a dictionary assigning each parameter of the problem the role
        # 'calibration' if it should be fitted or 'const' if it should not be
        # fitted; example: one entry could be self._prm_roles['a'] = 'const';
        # this dict is managed internally and should not be edited directly
        self._prm_roles = {}

        # the vector of all 'calibration'-parameters is called theta; here, each
        # parameter name is associated with the corresponding index in the
        # vector theta; the ordering will be defined by the order in which the
        # parameters are added to the problem; this dict is managed internally
        # # and should not be edited directly
        self._theta_dict = {}

        # this dictionary stores the values of all 'const'-parameters used in
        # the definition of the inference problem; this dict is managed
        # internally and should not be edited directly
        self._const_dict = {}

        # this dictionary is intended for storing the measured data from
        # experiments (see self.add_experiment); this dict is managed
        # internally and should not be edited directly
        self._experiments = {}

        #
        self.logprior_classes = {'normal': LogPriorNormal,
                                 'lognormal': LogPriorLognormal,
                                 'uniform': LogPriorUniform,
                                 'weibull': LogPriorWeibull}

        self._priors = {}

        self.model_errors = {}
        self.noise_models = {}

    def __str__(self):

        title_string = underlined_string(self.name, n_empty_start=2)

        n_prms = len(self._prm_names)
        prms_string = underlined_string("Parameter overview", symbol="-")
        prms_string += f"Number of parameters:   {n_prms}\n"
        for group in self._prm_names_dict.keys():
            prms_string += tcs(f'{group.capitalize()} parameters',
                               self._prm_names_dict[group])

        const_prms_str = underlined_string("Constant parameters", symbol="-")
        w = len(max(self._const_dict.keys(), key=len)) + 2
        for prm_name, prm_value in self._const_dict.items():
            const_prms_str += tcs(prm_name, f"{prm_value:.2f}", col_width=w)

        prior_str = underlined_string("Priors defined", symbol="-")
        w = len(max(self._priors.keys(), key=len)) + 2

        for prior_name, prior_obj in self._priors.items():
            prior_str += tcs(prior_name, str(prior_obj), col_width=w)

        full_string = title_string + prms_string + const_prms_str + prior_str

        return full_string

    def add_parameter(self, prm_name, prm_type, const=None, prior=None):

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

        # derive the parameter role from the 'const' argument
        prm_role = 'calibration' if const is None else 'const'

        # check whether the parameter name was used before; note that all
        # parameters (across types!) must have unique names
        if prm_name in self._prm_names:
            raise RuntimeError(
                f"A parameter with name '{prm_name}' has already been" +
                f" defined. Please choose another name."
            )

        # add the parameter to its type-class
        self._prm_names.append(prm_name)
        self._prm_names_dict[prm_type].append(prm_name)
        self._prm_names_dict[prm_role].append(prm_name)

        # in the case when the added parameter is a 'calibration'-parameter, it
        # needs to be added to the theta dictionary
        if prm_role == 'calibration':
            self._theta_dict[prm_name] = len(self._theta_dict)
        else:
            self._const_dict[prm_name] = const

        # add the prior internally
        if prior is not None:
            prior_type = prior[0]  # e.g. 'normal', 'lognormal', etc.
            prior_dict = prior[1]  # dictionary with parameter-value pairs
            prior_name = f"{prm_name}_{prior_type}"
            for prior_parameter_name, value in prior_dict.items():
                new_name = f"{prior_parameter_name}_{prm_name}"
                self.add_parameter(new_name, 'prior', const=value)
            self._add_prior(prior_name, prior_type, prior_dict, prm_name)

    def add_experiment(self, exp_name, exp_input, exp_output):
        self._experiments[exp_name] = {'input': {'sensor': exp_input[0],
                                                 'value': exp_input[1]},
                                       'output': {'sensor': exp_output[0],
                                                  'value': exp_output[1]}}

    def add_prior(self, name, cl):
        self.logprior_classes[name] = cl

    def _add_prior(self, prior_name, prior_type, prm_dict, ref_prm):
        self._priors[prior_name] = dc(self.logprior_classes[prior_type](prm_dict, ref_prm))

    def reduce_theta(self, theta, prm_names):
        if type(prm_names) is str:
            theta_red = theta[self._theta_dict[prm_names]]
        else:
            theta_red = [theta[self._theta_dict[prm_name]]
                         for prm_name in prm_names]
        return theta_red

    def add_model(self, model, prm_order):
        self.model = model
        self.prm_order = prm_order


    def evaluate_model(self, theta, experiments=None):

        if experiments is None:
            experiments = self._experiments

        theta_model = self.reduce_theta(theta, self.prm_order)

        model_predictions = {}

        for experiment_name, experiment_dict in experiments.items():
            x = experiment_dict['input']['value']
            value = self.model(x, theta_model)
            model_predictions[experiment_name] = value

        return model_predictions





    def add_model_error(self, model_error, ref=None):
        # define the reference and assert that it is not taken yet
        ref = ref or len(self.model_errors)
        assert ref not in self.model_errors
        # add the model error under the given/derived reference
        self.model_errors[ref] = model_error
        return ref

    def add_noise_model(self, noise_model, ref=None):
        # define the reference and assert that it is not taken yet
        ref = ref or len(self.noise_models)
        assert ref not in self.noise_models
        # add the model error under the given/derived reference
        self.noise_models[ref] = noise_model
        return ref

    def logprior(self, theta):

        lp = 0.0
        for prior in self._priors.values():
            x_prior = self.reduce_theta(theta, prior.ref_prm)
            lp += prior(x_prior)

        return lp

    def loglike(self, theta):

        ll = 0.0
        for ref, noise_model in self.noise_models.items():
            ll += noise_model.loglike_contribution(theta)









class ForwardModel:

    def __init__(self):
        self.prm_names_dict = {'model': [],
                               'latent': [],
                               'const': []}


class MyForwardModel(ForwardModel):
    def __init__(self):
        super().__init__()

    def __call__(self, x, theta):
        return theta[0]


# -----------------------------------------------------------------------------#

# the number of conducted experiments
n_tests = 100
a_true = 2.5
b_true = 1.7
sigma_noise = 0.1
seed = 1
show_data = False

# data-generation process; normal noise with constant variance around each point
np.random.seed(1)
x_test = np.linspace(0.0, 1.0, n_tests)
y_true = a_true * x_test + b_true
y_test = np.zeros(n_tests)
for i in range(n_tests):
    y_test[i] = np.random.normal(loc=y_true[i], scale=sigma_noise)

# plot the true and noisy data
if show_data:
    plt.scatter(x_test, y_test, label='measured data', s=10, c="red", zorder=10)
    plt.plot(x_test, y_true, label='true', c="black")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()

# initialize the inference problem with a useful name
problem = InferenceProblem("Linear model with normal noise")

# add all parameters to the problem
problem.add_parameter('a', 'model',
                      prior=('normal', {'loc': 2.0, 'scale': 1.0}))
problem.add_parameter('b', 'model',
                      prior=('lognormal', {'loc': 2.0, 'scale': 1.0}))
problem.add_parameter('prec', 'noise',
                      prior=('uniform', {'loc': 0.0, 'scale': 1.0}))

# define the forward model
def f(x, theta):
    a = theta[0]
    b = theta[1]
    return a * x + b


# add the forward model to the problem
problem.add_model(f, ['a', 'b'])

# add the experimental data
for i in range(n_tests):
    problem.add_experiment(f'Experiment_{i}',
                           ('x-Sensor', x_test[i]),
                           ('y-Sensor', y_test[i]))


print(problem)

print(problem.logprior([1,1,1]))

#mp = problem.evaluate_model([1., 1.])
#print(mp)


# problem.evaluate_model([200., 1., 1.])
#
#
# # problem.add_model(model, name='SimpleModel', prms=['E'], inp='Def-Sensor', out='E-Sensor')
# #
# #
# # def forward_model(theta, x):
# #     E =
# #     return x
# #
# # problem.add_model('Identify', forward_model, inp='Def-Sensor', out='E-Sensor')
#
#
#
#
# print('')
# print(problem.prm_names_dict)
# print('')
# print(problem.prm_names)
# print('')
# print(problem.theta_dict)
# print('')
# print(problem.const_dict)
# print('')
# print(problem.experiments)

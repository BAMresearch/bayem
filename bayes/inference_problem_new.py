# third party imports
import numpy as np


class InferenceProblem:

    def __init__(self, name):

        # this is the name of the problem
        self.name = name

        # this attributes is a list of all defined parameters in the problem
        self.prm_names = []

        # here the parameter names are associated with their respective type
        # from a perspective of the problem setup; a model-parameter is a
        # parameter of the forward model; a prior-parameter is a parameter
        # required to describe a prior distribution and a noise-parameter is a
        # parameter which describes some noise model in the problem definition;
        # all latent parameters are subject of the calibration and all const
        # parameters are not going to be fitted and considered constant instead
        self.prm_names_dict = {'model': [],
                               'prior': [],
                               'noise': [],
                               'latent': [],
                               'const': []}

        # this is a dictionary assigning each parameter of the problem the role
        # 'latent' if it should be fitted or 'const' if it should not be fitted;
        # for example, one entry could be self.prm_roles['E1'] = 'latent'
        self.prm_roles = {}

        # the vector of all latent parameters is called theta; here, each
        # parameter name is associated with the corresponding index in the
        # vector theta; the ordering will be defined by the order in which the
        # parameters are added to the problem
        self.theta_dict = {}

        # this dictionary stores the values of all constant parameters used in
        # the definition of the inference problem
        self.const_dict = {}

        self.experiments = {}

        self.model_errors = {}
        self.noise_models = {}

    def add_parameter(self, prm_name, prm_type, const=None):

        # derive the parameter role from the 'const' argument
        prm_role = 'latent' if const is None else 'const'

        # check whether the parameter name was used before; note that all
        # parameters (across types!) must have unique names
        if prm_name in self.prm_names:
            raise RuntimeError(
                f"A parameter with name '{prm_name}' has already been" +
                f" defined. Please choose another name."
            )

        # add the parameter to its type-class
        self.prm_names.append(prm_name)
        self.prm_names_dict[prm_type].append(prm_name)
        self.prm_names_dict[prm_role].append(prm_name)

        # in the case where the added parameter is a latent one, it needs to be
        # added to the theta dictionary
        if prm_role == 'latent':
            self.theta_dict[prm_name] = len(self.theta_dict)
        else:
            self.const_dict[prm_name] = const

    def add_experiment(self, exp_name, exp_input, exp_output):
        self.experiments[exp_name] = {'input': {'sensor': exp_input[0],
                                                'value': exp_input[1]},
                                      'output': {'sensor': exp_output[0],
                                                 'value': exp_output[1],
                                                 'prediction': None}}

    def reduce_theta(self, theta, prm_names):
        theta_red = [theta[self.theta_dict[prm_name]] for prm_name in prm_names]
        return theta_red

    def add_model(self, model, model_prms):
        self.model = model
        self.model_prms = model_prms


    def evaluate_model(self, theta, experiments=None):

        theta_model = self.reduce_theta(theta, self.model_prms)

        if experiments is None:
            experiments = self.experiments

        for experiment in experiments.values():
            x = experiment['input']['value']
            experiment['output']['prediction'] = self.model(x, theta_model)


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
n_tests = 5

# artificially generate the test data
test_data = np.random.normal(loc=210, scale=20.0, size=n_tests)

# initialize the inference problem with a useful name
problem = InferenceProblem("Estimate Young's modulus")

# add all parameters to the problem
problem.add_parameter('E', 'model')
problem.add_parameter('mu_E', 'prior')
problem.add_parameter('sd_E', 'prior')
problem.add_parameter('prec', 'noise', const=1.0)

for i, E in enumerate(test_data):
    problem.add_experiment(f'Exp{i}', ('Def-Sensor', 1.0), ('E-Sensor', E))

problem.add_model(MyForwardModel(), ['E'])
problem.evaluate_model([200., 1., 1.])


# problem.add_model(model, name='SimpleModel', prms=['E'], inp='Def-Sensor', out='E-Sensor')
#
#
# def forward_model(theta, x):
#     E =
#     return x
#
# problem.add_model('Identify', forward_model, inp='Def-Sensor', out='E-Sensor')




print('')
print(problem.prm_names_dict)
print('')
print(problem.prm_names)
print('')
print(problem.theta_dict)
print('')
print(problem.const_dict)
print('')
print(problem.experiments)

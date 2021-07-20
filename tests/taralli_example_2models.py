# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports
from bayes.forward_model import ModelTemplate
from bayes.noise_new import NormalNoise
from bayes.inference_problem_new import InferenceProblem
from taralli.parameter_estimation.base import EmceeParameterEstimator

# the number of conducted experiments
n_tests = 100
a_true = 2.5
alpha_true = 0.7
b_true = 1.7
sigma_noise = 0.15
n_walkers = 20
n_steps = 500
seed = 1
show_data = True
infer_noise_parameter = True

# data-generation process; normal noise with constant variance around each point
np.random.seed(1)
x_test = np.linspace(0.0, 1.0, n_tests)
y_linear_true = a_true * x_test + b_true
y_quadratic_true = alpha_true * x_test**2 + b_true
y_test_linear = np.zeros(n_tests)
y_test_quadratic = np.zeros(n_tests)
for i in range(n_tests):
    y_test_linear[i] = np.random.normal(loc=y_linear_true[i], scale=sigma_noise)
    y_test_quadratic[i] = np.random.normal(loc=y_quadratic_true[i],
                                           scale=sigma_noise)

# plot the true and noisy data
if show_data:
    plt.scatter(x_test, y_test_linear, label='measured data (linear)', s=10,
                c="red", zorder=10)
    plt.plot(x_test, y_linear_true, label='true (linear)', c="black")
    plt.scatter(x_test, y_test_quadratic, label='measured data (quadratic)',
                s=10, c="orange", zorder=10)
    plt.plot(x_test, y_quadratic_true, label='true (quadratic)', c="blue")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()

# initialize the inference problem with a useful name
problem = InferenceProblem("Linear model with normal noise")

# add all parameters to the problem
problem.add_parameter('a', 'model', info="Slope of the graph in linear model",
                      tex='$a$ (linear)',
                      prior=('normal', {'loc': 2.0, 'scale': 1.0}))
problem.add_parameter('alpha', 'model', info="Factor of quadratic term",
                      tex=r'$\alpha$ (quad.)',
                      prior=('normal', {'loc': 2.0, 'scale': 1.0}))
problem.add_parameter('b', 'model', info="Intersection of graph with y-axis",
                      tex='$b$ (shared)',
                      prior=('normal', {'loc': 1.0, 'scale': 1.0}))
problem.add_parameter('sigma', 'noise', tex=r"$\sigma$ (noise)",
                      info="Standard deviation of zero-mean noise model",
                      prior=('uniform', {'loc': 0.1, 'scale': 1.9}))

# define the linear forward model
class LinearModel(ModelTemplate):
    def __call__(self, x, prms):
        a = prms['a']
        b = prms['b']
        return a * x + b


# define the quadratic forward model
class QuadraticModel(ModelTemplate):
    def __call__(self, x, prms):
        alpha = prms['alpha']
        b = prms['beta']
        return alpha * x**2 + b


# add the forward model to the problem
problem.add_forward_model('LinearModel', LinearModel, ['a', 'b'])
problem.add_parameter_alias('b', 'beta')
problem.add_forward_model('QuadraticModel', QuadraticModel, ['alpha', 'beta'])


# add the experimental data
for i in range(n_tests):
    problem.add_experiment(f'Experiment_Linear_{i}',
                           ('x-Sensor', x_test[i]),
                           ('y-Sensor', y_test_linear[i]), 'LinearModel')
    problem.add_experiment(f'Experiment_Quadratic_{i}',
                           ('x-Sensor', x_test[i]),
                           ('y-Sensor', y_test_quadratic[i]), 'QuadraticModel')


# add the noise model to the problem
problem.add_noise_model('y-Sensor', NormalNoise, ['sigma'])

print(problem)

init_array = np.zeros((n_walkers, problem.n_calibration_prms))
init_array[:, 0] = a_true + np.random.randn(n_walkers)
init_array[:, 1] = alpha_true + np.random.randn(n_walkers)
init_array[:, 2] = b_true + np.random.randn(n_walkers)
init_array[:, 3] = np.random.uniform(0.05, 1.0, n_walkers)

emcee_model = EmceeParameterEstimator(
    log_likelihood=problem.loglike,
    log_prior=problem.logprior,
    ndim=problem.n_calibration_prms,
    nwalkers=n_walkers,
    sampling_initial_positions=init_array,
    nsteps=n_steps,
)

emcee_model.estimate_parameters()
emcee_model.plot_posterior(dim_labels=problem.get_theta_names(tex=True))
plt.show(block=True)

emcee_model.summary()

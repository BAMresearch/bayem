# ============================================================================ #
#                                   Imports                                    #
# ============================================================================ #

# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports
from bayes.forward_model import ModelTemplate
from bayes.forward_model import OutputSensor
from bayes.noise_new import NormalNoiseZeroMean
from bayes.inference_problem_new import InferenceProblem
from taralli.parameter_estimation.base import EmceeParameterEstimator

# ============================================================================ #
#                              Set numeric values                              #
# ============================================================================ #

# 'true' value of a, and its normal prior parameters
a_true = 2.5
loc_a = 2.0
scale_a = 1.0

# 'true' value of b, and its normal prior parameters
b_true = 1.7
loc_b = 1.0
scale_b = 1.0

# 'true' value of alpha, and its normal prior parameters
alpha_true = 0.7
loc_alpha = 2.0
scale_alpha = 1.0

# 'true' value of sigma, and its normal prior parameters
sigma_true = 0.15
low_sigma = 0.1
high_sigma = 2.0

# the number of generated experiments and seed for random numbers
n_tests = 100
seed = 1

# taralli settings
n_walkers = 20
n_steps = 1000

# ============================================================================ #
#                          Define the Forward Models                           #
# ============================================================================ #

class LinearModel(ModelTemplate):
    def response(self, inp, sensor):
        x = inp['x']
        a = inp['a']
        b = inp['b']
        return a * x + b

class QuadraticModel(ModelTemplate):
    def response(self, inp, sensor):
        x = inp['x']
        alpha = inp['alpha']
        beta = inp['beta']
        return alpha * x**2 + beta

# ============================================================================ #
#                         Define the Inference Problem                         #
# ============================================================================ #

# initialize the inference problem with a useful name
problem = InferenceProblem("Two models with shared parameter and normal noise")

# add all parameters to the problem
problem.add_parameter('a', 'model',
                      info="Slope of the graph in linear model",
                      tex='$a$ (linear)',
                      prior=('normal', {'loc': loc_a,
                                        'scale': scale_a}))
problem.add_parameter('alpha', 'model',
                      info="Factor of quadratic term",
                      tex=r'$\alpha$ (quad.)',
                      prior=('normal', {'loc': loc_alpha,
                                        'scale': scale_alpha}))
problem.add_parameter('b', 'model',
                      info="Intersection of graph with y-axis",
                      tex='$b$ (shared)',
                      prior=('normal', {'loc': loc_b,
                                        'scale': scale_b}))
problem.add_parameter('sigma', 'noise',
                      tex=r"$\sigma$ (noise)",
                      info="Standard deviation of zero-mean noise model",
                      prior=('uniform', {'low': low_sigma,
                                         'high': high_sigma}))

# this adds the alias 'beta' to parameter 'b'; just to show how it can be done;
# useful when the same parameter has different names in different models
problem.add_parameter_alias('b', 'beta')

# add the forward model to the problem
out = OutputSensor("y")
problem.add_forward_model("LinearModel",
                          LinearModel(['a', 'b'], [out]))
problem.add_forward_model('QuadraticModel',
                          QuadraticModel(['alpha', 'beta'], [out]))

# add the noise model to the problem
problem.add_noise_model('y', NormalNoiseZeroMean(['sigma']))

# ============================================================================ #
#                    Add test data to the Inference Problem                    #
# ============================================================================ #

# data-generation process; normal noise with constant variance around each point
np.random.seed(seed)
x_test = np.linspace(0.0, 1.0, n_tests)
y_linear_true = a_true * x_test + b_true
y_quadratic_true = alpha_true * x_test**2 + b_true
y_test_linear = np.zeros(n_tests)
y_test_quadratic = np.zeros(n_tests)
for i in range(n_tests):
    y_test_linear[i] = np.random.normal(loc=y_linear_true[i], scale=sigma_true)
    y_test_quadratic[i] = np.random.normal(loc=y_quadratic_true[i],
                                           scale=sigma_true)

# add the experimental data
for i in range(n_tests):
    problem.add_experiment(f'Experiment_Linear_{i}',
                           exp_input={'x': x_test[i]},
                           exp_output={'y': y_test_linear[i]},
                           fwd_model_name='LinearModel')
    problem.add_experiment(f'Experiment_Quadratic_{i}',
                           exp_input={'x': x_test[i]},
                           exp_output={'y': y_test_quadratic[i]},
                           fwd_model_name='QuadraticModel')

# give problem overview
print(problem)

# plot the true and noisy data
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
plt.draw()  # does not stop execution

# ============================================================================ #
#                          Solve problem with Taralli                          #
# ============================================================================ #

# provide initial samples
init_array = np.zeros((n_walkers, problem.n_calibration_prms))
init_array[:, 0] = np.random.normal(loc_a, scale_a, n_walkers)
init_array[:, 1] = np.random.normal(loc_alpha, scale_alpha, n_walkers)
init_array[:, 2] = np.random.normal(loc_b, scale_b, n_walkers)
init_array[:, 3] = np.random.uniform(low_sigma, high_sigma, n_walkers)

# set up sampling task
emcee_model = EmceeParameterEstimator(
    log_likelihood=problem.loglike,
    log_prior=problem.logprior,
    ndim=problem.n_calibration_prms,
    nwalkers=n_walkers,
    sampling_initial_positions=init_array,
    nsteps=n_steps,
)

# perform sampling
emcee_model.estimate_parameters()

# plot the results
emcee_model.plot_posterior(dim_labels=problem.get_theta_names(tex=True))
plt.show(block=True)

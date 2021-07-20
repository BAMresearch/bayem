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
b_true = 1.7
sigma_noise = 0.5
n_walkers = 20
n_steps = 50
seed = 1
show_data = False
infer_noise_parameter = True

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
problem.add_parameter('a', 'model', info="Slope of the graph", tex="$a$",
                      prior=('normal', {'loc': 2.0, 'scale': 1.0}))
problem.add_parameter('b', 'model', info="Intersection of graph with y-axis",
                      tex='$b$', prior=('normal', {'loc': 1.0, 'scale': 1.0}))
problem.add_parameter('sigma', 'noise', tex=r"$\sigma$",
                      info="Standard deviation of zero-mean noise model",
                      prior=('uniform', {'loc': 0.1, 'scale': 1.9}))

# overwrite default info-strings for a's prior parameters
problem.change_parameter_info("loc_a", "Mean of normal prior for 'a'")
problem.change_parameter_info("scale_a",
                              "Standard deviation of normal prior for 'a'")

# in case the 'prec' should not be inferred, change it to a constant
if not infer_noise_parameter:
    problem.change_parameter_role('prec', const=1.0)
    problem.change_parameter_role('b', const=1.7)
    # problem.change_parameter_role('b', prior=('normal', {'loc': 2.0, 'scale': 1.0}))

# define the forward model
class LinearModel(ModelTemplate):
    def __call__(self, x, prms):
        a = prms['a']
        b = prms['b']
        return a * x + b


# add the forward model to the problem
problem.add_forward_model("LinearModel", LinearModel, ['a', 'b'])

# add the experimental data
for i in range(n_tests):
    problem.add_experiment(f'Experiment_{i}',
                           ('x-Sensor', x_test[i]),
                           ('y-Sensor', y_test[i]),
                           "LinearModel")

# add the noise model to the problem
problem.add_noise_model('y-Sensor', NormalNoise, ['sigma'])

problem.theta_explanation()

print(problem)

init_array = np.zeros((n_walkers, problem.n_calibration_prms))
init_array[:, 0] = a_true + np.random.randn(n_walkers)
#if problem.n_calibration_prms == 2:
init_array[:, 1] = b_true + np.random.randn(n_walkers)
init_array[:, 2] = np.random.uniform(0.1, 2.0, n_walkers)
print(init_array[:, 2])

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

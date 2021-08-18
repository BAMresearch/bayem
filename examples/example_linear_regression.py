"""
Simple linear regression example with two model and one noise parameter
--------------------------------------------------------------------------------
The model equation is y = a * x + b with a, b being the model parameters and the
noise model is a normal zero-mean distribution with the std. deviation to infer.
The problem is solved via sampling using taralli.
"""
# ============================================================================ #
#                                   Imports                                    #
# ============================================================================ #

# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports
from bayes.forward_model import ModelTemplate
from bayes.forward_model import OutputSensor
from bayes.noise import NormalNoiseZeroMean
from bayes.inference_problem import InferenceProblem
from bayes.solver import taralli_solver

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

# 'true' value of noise sd, and its uniform prior parameters
sigma_noise = 0.5
low_sigma = 0.1
high_sigma = 0.8

# the number of generated experiments and seed for random numbers
n_tests = 50
seed = 1

# taralli settings
n_walkers = 20
n_steps = 1000

# ============================================================================ #
#                           Define the Forward Model                           #
# ============================================================================ #

class LinearModel(ModelTemplate):
    def response(self, inp, sensor):
        x = inp['x']
        a = inp['m']
        b = inp['b']
        return a * x + b

# ============================================================================ #
#                         Define the Inference Problem                         #
# ============================================================================ #

# initialize the inference problem with a useful name
problem = InferenceProblem("Linear regression with normal noise")

# add all parameters to the problem
problem.add_parameter('a', 'model',
                      info="Slope of the graph", tex="$a$",
                      prior=('normal', {'loc': loc_a, 'scale': scale_a}))
problem.add_parameter('b', 'model',
                      info="Intersection of graph with y-axis", tex='$b$',
                      prior=('normal', {'loc': loc_b, 'scale': scale_b}))
problem.add_parameter('sigma', 'noise',
                      info="Std. dev, of 0-mean noise model", tex=r"$\sigma$",
                      prior=('uniform', {'low': low_sigma, 'high': high_sigma}))

# add the forward model to the problem
out = OutputSensor("y")
problem.add_forward_model("LinearModel", LinearModel([{'a': 'm'}, 'b'], [out]))

# add the noise model to the problem
problem.add_noise_model(out.name, NormalNoiseZeroMean(['sigma']))

# ============================================================================ #
#                    Add test data to the Inference Problem                    #
# ============================================================================ #

# data-generation process; normal noise with constant variance around each point
np.random.seed(seed)
x_test = np.linspace(0.0, 1.0, n_tests)
y_true = a_true * x_test + b_true
y_test = np.zeros(n_tests)
for i in range(n_tests):
    y_test[i] = np.random.normal(loc=y_true[i], scale=sigma_noise)

# add the experimental data
for i in range(n_tests):
    problem.add_experiment(f'Experiment_{i}',
                           exp_input={'x': x_test[i]},
                           exp_output={'y': y_test[i]},
                           fwd_model_name="LinearModel")

# give problem overview
problem.info()

# plot the true and noisy data
plt.scatter(x_test, y_test, label='measured data', s=10, c="red", zorder=10)
plt.plot(x_test, y_true, label='true', c="black")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.draw()  # does not stop execution

# ============================================================================ #
#                          Solve problem with Taralli                          #
# ============================================================================ #

# code is bundled in a specific solver routine
taralli_solver(problem, n_walkers=n_walkers, n_steps=n_steps)
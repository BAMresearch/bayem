"""
Calibration of exponential decay model with sensor error
--------------------------------------------------------------------------------
The model equation is y = m0 * exp(-k * t) with k being the model parameter, m0
is the initial measured mass, and t is the time. Experiments are conducted with
two different mass sensors, each one associated with an own normal noise model.
Independent of the used mass sensor, a clock is used with an unknown additive
bias (the sensor error). This bias, called delta_t here, is inferred too. The
problem is solved via sampling using taralli.
"""
# ============================================================================ #
#                                   Imports                                    #
# ============================================================================ #

# third party imports
import numpy as np

# local imports
from bayes.forward_model import ModelTemplate
from bayes.forward_model import OutputSensor
from bayes.noise import NormalNoiseZeroMean
from bayes.inference_problem import InferenceProblem
from bayes.solver import taralli_solver

# ============================================================================ #
#                              Set numeric values                              #
# ============================================================================ #

# 'true' value of k (decay parameter), and its uniform prior parameters
k_true = 1.0
low_k = 0.7
high_k = 2.0

# 'true' value of noise_1 sd, and its uniform prior parameters
sigma1_noise = 0.5
low_sigma1 = 0.1
high_sigma1 = 0.7

# 'true' value of noise_2 sd, and its uniform prior parameters
sigma2_noise = 0.2
low_sigma2 = 0.1
high_sigma2 = 0.7

# 'true' value of additive clock error, and its error bound
clock_error_true = 0.3
clock_error_bound = 1.0

# the number of generated experiments and seed for random numbers
n_m0_1 = 5  # number of start masses for mass sensor 1
n_t_1 = 4  # number of measurement times per start mass for mass sensor 1
n_m0_2 = 5  # number of start masses for mass sensor 2
n_t_2 = 4  # number of measurement times per start mass for mass sensor 2
seed = 1

# taralli settings
n_walkers = 20
n_steps = 1000

# ============================================================================ #
#                           Define the Forward Model                           #
# ============================================================================ #

class ExponentialDecay(ModelTemplate):
    def response(self, inp, sensor):
        t = inp['t'] + clock_error_true
        m0 = inp['m0']
        k = inp['k']
        return m0 * np.exp(-k * t)

# ============================================================================ #
#                         Define the Inference Problem                         #
# ============================================================================ #

# initialize the inference problem with a useful name
problem = InferenceProblem("Exponential decay model with sensor error")

# add all parameters to the problem
problem.add_parameter('k', 'model',
                      tex="$k$",
                      info="Decay constant",
                      prior=('uniform', {'low': low_k,
                                         'high': high_k}))
problem.add_parameter('sigma_1', 'noise',
                      tex=r"$\sigma_1$",
                      info="Standard deviation of zero-mean noise_1 model",
                      prior=('uniform', {'low': low_sigma1,
                                         'high': high_sigma1}))
problem.add_parameter('sigma_2', 'noise',
                      tex=r"$\sigma_2$",
                      info="Standard deviation of zero-mean noise_2 model",
                      prior=('uniform', {'low': low_sigma2,
                                         'high': high_sigma2}))

# add the forward model to the problem
ms_1 = OutputSensor("MassSensor_1")
problem.add_forward_model("ExponentialDecay_MassSensor_1",
                          ExponentialDecay(['k'], [ms_1]))
ms_2 = OutputSensor("MassSensor_2")
problem.add_forward_model("ExponentialDecay_MassSensor_2",
                          ExponentialDecay(['k'], [ms_2]))

# add the noise model to the problem
problem.add_noise_model(ms_1.name, NormalNoiseZeroMean(['sigma_1']))
problem.add_noise_model(ms_2.name, NormalNoiseZeroMean(['sigma_2']))

# add a sensor error to the time measurement
problem.add_sensor_error(['t'], name="delta_t", plusminus=clock_error_bound,
                         tex=r"$\Delta_t$",  error_type="abs")

# ============================================================================ #
#                    Add test data to the Inference Problem                    #
# ============================================================================ #

# data-generation process; normal noise with constant variance around each point
np.random.seed(seed)
m0_test = np.random.uniform(1.0, 10.0, n_m0_1)
t_test = np.linspace(1.0, 5.0, n_t_1)
for i in range(n_m0_1):
    for j in range(n_t_1):
        mt = m0_test[i] * np.exp(-k_true * t_test[j]) +\
             np.random.normal(0, sigma1_noise)
        problem.add_experiment(f'Experiment1_{i}_{j}',
                               exp_input={'t': t_test[j],
                                          'm0': m0_test[i]},
                               exp_output={'MassSensor_1': mt},
                               fwd_model_name="ExponentialDecay_MassSensor_1")
m0_test = np.random.uniform(1.0, 10.0, n_m0_2)
t_test = np.linspace(1.0, 5.0, n_t_2)
for i in range(n_m0_2):
    for j in range(n_t_2):
        mt = m0_test[i] * np.exp(-k_true * t_test[j]) +\
             np.random.normal(0, sigma2_noise)
        problem.add_experiment(f'Experiment2_{i}_{j}',
                               exp_input={'t': t_test[j],
                                          'm0': m0_test[i]},
                               exp_output={'MassSensor_2': mt},
                               fwd_model_name="ExponentialDecay_MassSensor_2")

# give problem overview
problem.info()

# ============================================================================ #
#                          Solve problem with Taralli                          #
# ============================================================================ #

# code is bundled in a specific solver routine
taralli_solver(problem, n_walkers=n_walkers, n_steps=n_steps)

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
    def __call__(self, inp, prms):
        t = inp['t'] + clock_error_true
        m0 = inp['m0']
        k = prms['k']
        response = {}
        for out_sens in self.output_sensors:
            response[out_sens.name] = m0 * np.exp(-k * t)
        return response

# ============================================================================ #
#                         Define the Inference Problem                         #
# ============================================================================ #

# initialize the inference problem with a useful name
problem = InferenceProblem("Linear model with normal noise and prior-prior")

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

# ============================================================================ #
#                     Add an additive sensor error for 't'                     #
# ============================================================================ #

# add a sensor error to the time measurement
problem.add_sensor_error(['t'], name="delta_t", plusminus=clock_error_bound,
                         tex=r"$\Delta_t$",  error_type="absolute")

# give problem overview
print(problem)

# ============================================================================ #
#                          Solve problem with Taralli                          #
# ============================================================================ #

# provide initial samples
init_array = np.zeros((n_walkers, problem.n_calibration_prms))
init_array[:, 0] = np.random.uniform(low_k, high_k, n_walkers)
init_array[:, 1] = np.random.uniform(low_sigma1, high_sigma1, n_walkers)
init_array[:, 2] = np.random.uniform(low_sigma2, high_sigma2, n_walkers)
init_array[:, 3] = np.random.uniform(-clock_error_bound, clock_error_bound,
                                     n_walkers)

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

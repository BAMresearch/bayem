# ============================================================================ #
#                                   Imports                                    #
# ============================================================================ #

# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports
from bayes.forward_model import ModelTemplate
from bayes.forward_model import OutputSensor
from bayes.inference_problem_new import InferenceProblem
from bayes.noise_new import NormalNoiseZeroMean
from taralli.parameter_estimation.base import EmceeParameterEstimator

# ============================================================================ #
#                              Set numeric values                              #
# ============================================================================ #

# 'true' value of A, and its normal prior parameters
A_true = 42.0
loc_A = 40.0
scale_A = 5.0

# 'true' value of B, and its normal prior parameters
B_true = 6174.0
loc_B = 6000.0
scale_B = 300.0

# 'true' value of sd_S1, and its uniform prior parameters
sd_S1_true = 0.2
low_S1 = 0.1
high_S1 = 0.7

# 'true' value of sd_S2, and its uniform prior parameters
sd_S2_true = 0.4
low_S2 = 0.1
high_S2 = 0.7

# 'true' value of sd_S3, and its uniform prior parameters
sd_S3_true = 0.6
low_S3 = 0.1
high_S3 = 0.7

# define sensor positions
pos_s1 = 0.2
pos_s2 = 0.5
pos_s3 = 42.0

# taralli settings
n_walkers = 20
n_steps = 1000

# ============================================================================ #
#                           Define the Forward Model                           #
# ============================================================================ #

class PositionSensor(OutputSensor):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position

class LinearModel(ModelTemplate):
    def __call__(self, inp, prms):
        t = inp['time']
        A = prms['A']
        B = prms['B']
        response = {}
        for out in self.output_sensors:
            response[out.name] = A * out.position + B * t
        return response

# ============================================================================ #
#                         Define the Inference Problem                         #
# ============================================================================ #

# initialize the inference problem with a useful name
problem = InferenceProblem("Linear model with normal noise")

# add all parameters to the problem
problem.add_parameter('A', 'model',
                      prior=('normal', {'loc': loc_A, 'scale': scale_A}),
                      info="Slope of the graph",
                      tex="$A$")
problem.add_parameter('B', 'model',
                      prior=('normal', {'loc': loc_A, 'scale': loc_B}),
                      info="Intersection of graph with y-axis",
                      tex='$B$')
problem.add_parameter('sigma_1', 'noise',
                      prior=('uniform', {'low': low_S1, 'high': high_S1}),
                      info="Std. dev. of zero-mean noise model for S1",
                      tex=r"$\sigma_1$")
problem.add_parameter('sigma_2', 'noise',
                      prior=('uniform', {'low': low_S2, 'high': high_S2}),
                      info="Std. dev. of zero-mean noise model for S1",
                      tex=r"$\sigma_2$")
problem.add_parameter('sigma_3', 'noise',
                      prior=('uniform', {'low': low_S3, 'high': high_S3}),
                      info="Std. dev. of zero-mean noise model for S1",
                      tex=r"$\sigma_3$")

# add the forward model to the problem
out_1 = PositionSensor("S1", pos_s1)
out_2 = PositionSensor("S2", pos_s2)
out_3 = PositionSensor("S3", pos_s3)
linear_model = LinearModel(['A', 'B'], [out_1, out_2, out_3])
problem.add_forward_model("LinearModel", linear_model)

# add the noise model to the problem
problem.add_noise_model(out_1.name, NormalNoiseZeroMean(['sigma_1']))
problem.add_noise_model(out_2.name, NormalNoiseZeroMean(['sigma_2']))
problem.add_noise_model(out_3.name, NormalNoiseZeroMean(['sigma_3']))

# ============================================================================ #
#                    Add test data to the Inference Problem                    #
# ============================================================================ #

# add the experimental data
sd_dict = {out_1.name: sd_S1_true,
           out_2.name: sd_S2_true,
           out_3.name: sd_S3_true}

def generate_data(n_time_steps, n=None):
    time_steps = np.linspace(0, 1, n_time_steps)
    for t in time_steps:
        inp = {'time': t}
        prms = {'A': A_true, 'B': B_true}
        model_res = linear_model(inp, prms)
        for key, val in model_res.items():
            model_res[key] = val + np.random.normal(0.0, sd_dict[key])
        problem.add_experiment(f'Experiment{n}_t{t:.3f}',
                               exp_input=inp, exp_output=model_res,
                               fwd_model_name='LinearModel')
for n_exp, n_t in enumerate([101, 51]):
    generate_data(n_t, n=n_exp)

# give problem overview
print(problem)

# ============================================================================ #
#                          Solve problem with Taralli                          #
# ============================================================================ #

# provide initial samples
init_array = np.zeros((n_walkers, problem.n_calibration_prms))
init_array[:, 0] = np.random.normal(loc_A, scale_A, n_walkers)
init_array[:, 1] = np.random.normal(loc_B, scale_B, n_walkers)
init_array[:, 2] = np.random.uniform(low_S1, high_S1, n_walkers)
init_array[:, 3] = np.random.uniform(low_S2, high_S2, n_walkers)
init_array[:, 4] = np.random.uniform(low_S3, high_S3, n_walkers)

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

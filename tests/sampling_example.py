import unittest
import numpy as np

from bayes.parameters import ParameterList
from bayes.inference_problem import VariationalBayesProblem, ModelErrorInterface
from bayes.noise import UncorrelatedNoiseModel

from bayes.noise import UncorrelatedNoiseTerm

from pyro.params.param_store import ParamStoreDict
from scipy.stats import norm
"""
Not really a test yet.

At the bottom it demonstrates how to infer the VariationalBayesProblem
with pymc3.

"""


class Sensor:
    def __init__(self, name, shape=(1, 1)):
        self.name = name
        self.shape = shape

    def __repr__(self):
        return f"{self.name} {self.shape}"


class MySensor(Sensor):
    def __init__(self, name, position):
        super().__init__(name, shape=(1, 1))
        self.position = position


class MyForwardModel:
    def __call__(self, parameter_list, sensors, time_steps):
        """
        evaluates 
            fw(x, t) = A * x + B * t
        """
        A = parameter_list["A"]
        B = parameter_list["B"]

        result = {}
        for sensor in sensors:
            result[sensor] = A * sensor.position + B * time_steps
        return result

    def parameter_list(self):
        p = ParameterList()
        p.define("A", None)
        p.define("B", None)
        return p


class MyModelError(ModelErrorInterface):
    def __init__(self, fw, data):
        self._fw = fw
        self._ts, self._sensor_data = data
        self.parameter_list = fw.parameter_list()

    def __call__(self):
        sensors = list(self._sensor_data.keys())
        model_response = self._fw(self.parameter_list, sensors, self._ts)
        error = {}
        for sensor in sensors:
            error[sensor] = model_response[sensor] - self._sensor_data[sensor]
        return error

# this is not the final version (we should move the definitions of prior etc to InferenceProblem)
class PytorchTaralliProblem(VariationalBayesProblem):
    def __init__(self):
        super().__init__()

    def posterior_pytorch_model(self):
        sampled_parameters = ParamStoreDict()
        for name, (mean, sd) in self.prm_prior.items():
            # idx = problem.latent[name].start_idx
            assert self.latent[name].N == 1  # vector parameters not yet supported!
            sampled_parameters = pyro.sample(name, dist.Normal(mean, sd))

        for name, gamma in self.noise_prior.items():
            # idx = problem.latent[name].start_idx
            shape, scale = gamma.shape, gamma.scale
            alpha, beta = shape, 1.0 / scale
            # check for Gamma and Inverse Gamma
            sampled_noise_precision = pyro.sample(name, dist.Gamma(alpha, beta))
            self.noise_models[name].parameter_list['precision'] = sampled_noise_precision

        for noise_key, noise_term in self.noise_models.items():
            pyro.sample(noise_key + "_latent",
                        dist.Normal(self.wrapper_function(sampled_parameters, noise_key),
                        noise_term.parameter_list['precision']),
                        obs=torch.zeros(len(vector)))
        return

    def wrapper_function(self, sampled_parameters, noise_key):
        for name, value in sampled_parameters.items():
            self.latent[name].set_value(value)

        # compute raw model error using the samples parameters, this is now done for each noise term separately
        raw_me = {}
        for key, me in self.model_errors.items():
            raw_me[key] = me()

        vector = self.noise_models[noise_key].vector_contribution(raw_me)
        return torch.from_numpy(vector)

    def logprior(self, parameter_vector):
        self.latent.update(parameter_vector)
        sum_log = 0.
        for name, (mean, sd) in self.prm_prior.items():
            assert self.latent[name].N == 1  # vector parameters not yet supported!
            sum_log +=  norm.logpdf(x=self.latent[name].value(parameter_vector), loc=mean, scale=sd)
        return sum_log

"""
###############################################################################

                            MAIN CODE

###############################################################################
"""

if __name__ == "__main__":
    # Define the sensor
    s1, s2, s3 = MySensor("S1", 0.2), MySensor("S2", 0.5), MySensor("S3", 42.0)

    fw = MyForwardModel()
    prm = fw.parameter_list()

    # set the correct values
    A_correct = 42.0
    B_correct = 6174.0
    prm["A"] = A_correct
    prm["B"] = B_correct

    np.random.seed(6174)
    noise_sd1 = 0.2
    noise_sd2 = 0.4

    def generate_data(N_time_steps, noise_sd):
        time_steps = np.linspace(0, 1, N_time_steps)
        model_response = fw(prm, [s1, s2, s3], time_steps)
        sensor_data = {}
        for sensor, perfect_data in model_response.items():
            sensor_data[sensor] = perfect_data + np.random.normal(
                0.0, noise_sd, N_time_steps
            )
        return time_steps, sensor_data

    data1 = generate_data(101, noise_sd1)
    data2 = generate_data(51, noise_sd2)

    me1 = MyModelError(fw, data1)
    me2 = MyModelError(fw, data2)

    problem = PytorchTaralliProblem()
    key1 = problem.add_model_error(me1)
    key2 = problem.add_model_error(me2)

    problem.latent["A"].add(me1.parameter_list, "A")
    problem.latent["A"].add(me2.parameter_list, "A")
    # or simply
    problem.define_shared_latent_parameter_by_name("B")

    problem.set_normal_prior("A", 40.0, 5.0)
    problem.set_normal_prior("B", 6000.0, 300.0)

    noise1 = UncorrelatedNoiseModel()
    noise1.add(key1, s1)
    noise1.add(key1, s2)
    noise1.add(key1, s3)

    noise2 = UncorrelatedNoiseModel()
    noise2.add(key2, s1)
    noise2.add(key2, s2)
    noise2.add(key2, s3)

    noise_key1 = problem.add_noise_model(noise1)
    noise_key2 = problem.add_noise_model(noise2)

    problem.set_noise_prior(noise_key1, 3 * noise_sd1, sd_shape=0.5)
    problem.set_noise_prior(noise_key2, 3 * noise_sd2, sd_shape=0.5)

    compute_linearized_VB = False
    compute_pyro = False
    compute_pymc3 = False
    compute_taralli = True

    if compute_linearized_VB:
        info = problem.run()
        print(info)

    if compute_pyro:
        import arviz as az
        import torch
        import pyro
        import pyro.distributions as dist
        import pyro.poutine as poutine
        from pyro.infer import MCMC, NUTS, Predictive

        nuts_kernel = NUTS(problem.posterior_pytorch_model, jit_compile=True)
        mcmc = MCMC(nuts_kernel,
                    num_samples=300,
                    warmup_steps=50,
                    num_chains=4
                    )
        mcmc.run()
        mcmc.summary(prob=0.5)

        posterior_samples = mcmc.get_samples()
        posterior_predictive = Predictive(problem.posterior_pytorch_model, posterior_samples)()
        prior = Predictive(problem.posterior_pytorch_model, num_samples=500)()

        pm_data = az.from_pyro(
            mcmc,
            prior=prior,
        )

        fig, axes = plt.subplots(1, 3, figsize=(11, 5), squeeze=False)
        az.plot_dist_comparison(pm_data, var_names=["E", "noise0"], figsize=(8, 9))
        az.plot_trace(pm_data, figsize=(11, 4))
        az.plot_pair(pm_data, kind="scatter", ax=axes[0, 0])
        # az.plot_posterior(pm_data, kind="hist")
        az.plot_posterior(pm_data, var_names="E", kind="hist", ax=axes[0, 1])
        az.plot_posterior(pm_data, var_names="noise0", kind="hist", ax=axes[0, 2])
        plt.show()

    if compute_pymc3:
        """
        We now transform the vb problem into a sampling problem. 
    
    
        1)  Set the parameters of the noise models latent. For convenience (since
            there is only one parameter per noise model), we define the global 
            name of the latent parameter to be the noise_key.
        """
        for noise_key in problem.noise_prior:
            noise_parameters = problem.noise_models[noise_key].parameter_list
            problem.latent[noise_key].add(noise_parameters, "precision")

        """
        2)  Wrap problem.loglike for a tool of your choice
        """
        import theano.tensor as tt

        class LogLike(tt.Op):
            itypes = [tt.dvector]  # expects a vector of parameter values when called
            otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

            def __init__(self, loglike):
                self.likelihood = loglike

            def perform(self, node, inputs, outputs):
                (theta,) = inputs  # this will contain my variables
                result = self.likelihood(theta)
                outputs[0][0] = np.array(result)  # output the log-likelihood

        pymc3_log_like = LogLike(problem.loglike)

        """
        3)  Define prior distributions in a tool of your choice!
        """
        import pymc3 as pm

        pymc3_prior = [None] * len(problem.latent)

        model = pm.Model()
        with model:
            for name, (mean, sd) in problem.prm_prior.items():
                idx = problem.latent[name].start_idx
                assert problem.latent[name].N == 1  # vector parameters not yet supported!
                pymc3_prior[idx] = pm.Normal(name, mu=mean, sigma=sd)

            for name, gamma in problem.noise_prior.items():
                idx = problem.latent[name].start_idx
                shape, scale = gamma.shape, gamma.scale
                alpha, beta = shape, 1.0 / scale
                pymc3_prior[idx] = pm.Gamma(name, alpha=alpha, beta=beta)

        """
        4)  Go!
        """
        with model:
            theta = tt.as_tensor_variable(pymc3_prior)
            pm.Potential("likelihood", pymc3_log_like(theta))

            trace = pm.sample(
                draws=1000,
                step=pm.Metropolis(),
                chains=4,
                tune=100,
                discard_tuned_samples=True,
            )

        summary = pm.summary(trace)
        print(summary)

        print(1.0 / info.noise[noise_key1].mean ** 0.5, 1.0 / info.noise[noise_key2].mean ** 0.5)

        means = summary["mean"]
        print(1.0 / means[noise_key1] ** 0.5, 1.0 / means[noise_key2] ** 0.5)

    if compute_taralli:
        from taralli.parameter_estimation.base import EmceeParameterEstimator

        noise1.parameter_list['precision'] = 1./noise_sd1**2 
        noise2.parameter_list['precision'] = 1./noise_sd2**2

        emcee_model = EmceeParameterEstimator(
            log_likelihood=problem.loglike,
            log_prior=problem.logprior,
            ndim=sum(latent_var.N for latent_var in problem.latent.values()),  #better create a function for that
            nwalkers=20,
            sampling_initial_positions=np.random.multivariate_normal([
                problem.prm_prior['A'][0], problem.prm_prior['B'][0]],
                [[problem.prm_prior['A'][1], 0], [0, problem.prm_prior['B'][1]]], 20), # improve accessing prior
            # mean/std
            nsteps=5000,
        )
        emcee_model.estimate_parameters()

        emcee_model.plot_posterior()
        emcee_model.summary()

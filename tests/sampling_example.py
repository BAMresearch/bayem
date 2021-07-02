import unittest
import numpy as np
import scipy.stats

from bayes.parameters import ParameterList
from bayes.inference_problem import (
    InferenceProblem,
    VariationalBayesSolver,
    ModelErrorInterface,
    TaralliSolver,
    gamma_from_sd,
)
from bayes.noise import UncorrelatedSensorNoise

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
    noise_sd = {s1: 0.2, s2: 0.4, s3: 0.6}

    def generate_data(N_time_steps):
        time_steps = np.linspace(0, 1, N_time_steps)
        model_response = fw(prm, [s1, s2, s3], time_steps)
        sensor_data = {}
        for sensor, perfect_data in model_response.items():
            sensor_data[sensor] = perfect_data + np.random.normal(
                0.0, noise_sd[sensor], N_time_steps
            )
        return time_steps, sensor_data

    data1 = generate_data(101)
    data2 = generate_data(51)

    me1 = MyModelError(fw, data1)
    me2 = MyModelError(fw, data2)

    problem = InferenceProblem()
    key1 = problem.add_model_error(me1)
    key2 = problem.add_model_error(me2)

    problem.latent["A"].add(me1.parameter_list, "A")
    problem.latent["A"].add(me2.parameter_list, "A")
    # or simply
    problem.define_shared_latent_parameter_by_name("B")

    problem.set_prior("A", scipy.stats.norm(40.0, 5.0))
    problem.set_prior("B", scipy.stats.norm(6000.0, 300.0))

    for sensor in [s1, s2, s3]:
        noise_model = UncorrelatedSensorNoise([sensor])
        noise_key = "noise" + sensor.name
        problem.add_noise_model(noise_model, noise_key)
        problem.latent[noise_key].add(noise_model.parameter_list, "precision")
        problem.set_prior(noise_key, gamma_from_sd(3 * noise_sd[sensor], shape=0.5))

    vb = VariationalBayesSolver(problem)
    info = vb.run()
    print(info)

    if True:
        
        import taralli.parameter_estimation.base as taralli_estimator
        taralli_solver = TaralliSolver(problem)
        nwalker = 20
        init = np.empty((nwalker, len(taralli_solver.prior)))
        for i, prior in enumerate(taralli_solver.prior.values()):
            init[:,i] = prior.rvs(nwalker)

        emcee = taralli_estimator.EmceeParameterEstimator(
                seed=6174,
                ndim=init.shape[1],
                nwalkers=init.shape[0],
                sampling_initial_positions=init,
                log_prior=taralli_solver.logprior,
                log_likelihood=taralli_solver.loglike
                )
        emcee.estimate_parameters(vectorize=True)
        emcee.summary()

        sampler = taralli_estimator.NestleParameterEstimator(
                seed=6174,
                ndim=len(problem.prior),
                log_likelihood=taralli_solver.loglike,
                prior_transform=taralli_solver.prior_transform
                )
        sampler.estimate_parameters()
        sampler.summary()

    if False:
    
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

        import pymc3 as pm

        pymc3_prior = [None] * len(problem.latent)

        model = pm.Model()
        with model:
            for name, latent in problem.latent.items():
                assert latent.N == 1  # vector parameters not yet supported!
                idx = latent.start_idx

                prior = problem.prior[name]
                if prior.dist.name == "norm":
                    pymc3_prior[idx] = pm.Normal(name, mu=prior.mean(), sigma=prior.std())
                elif prior.dist.name == "gamma":
                    scale = prior.var() / prior.mean()
                    shape = prior.mean() / scale
                    pymc3_prior[idx] = pm.Gamma(name, alpha=shape, beta=1./scale)

        with model:
            theta = tt.as_tensor_variable(pymc3_prior)
            pm.Potential("likelihood", pymc3_log_like(theta))

            trace = pm.sample(
                draws=2000,
                step=pm.Metropolis(),
                chains=4,
                tune=500,
                discard_tuned_samples=True,
            )

            summary = pm.summary(trace)

        print(summary)

        means = summary["mean"]
        for s in [s1, s2, s3]:
            print(s, 1./means["noise"+s.name]**0.5)

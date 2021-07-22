import unittest
import numpy as np

from bayes.parameters import ParameterList
from bayes.inference_problem import (
    VariationalBayesSolver,
    InferenceProblem,
)
from bayes.noise import UncorrelatedSensorNoise
import bayes.vb

"""
Not really a test yet.

At the bottom it demonstrates how to infer the VariationalBayesProblem
with pymc3.

"""


class MySensor:
    def __init__(self, name, position):
        self.name = name
        self.position = position


def my_forward_model(parameter_list, sensors, time_steps):
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


class MyModelError:
    def __init__(self, fw, data):
        self._fw = fw
        self._ts, self._sensor_data = data

    def __call__(self, parameter_list):
        sensors = list(self._sensor_data.keys())
        model_response = self._fw(parameter_list, sensors, self._ts)
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

    prm = ParameterList()

    # set the correct values
    A_correct = 42.0
    B_correct = 6174.0
    prm.define("A", A_correct)
    prm.define("B", B_correct)

    noise_sds = {s1: 0.2, s2: 0.4, s3: 0.6}
    np.random.seed(6174)

    def generate_data(N_time_steps):
        time_steps = np.linspace(0, 1, N_time_steps)
        model_response = my_forward_model(prm, [s1, s2, s3], time_steps)

        sensor_data = {}
        for sensor, perfect_data in model_response.items():
            sensor_data[sensor] = perfect_data + np.random.normal(
                0.0, noise_sds[sensor], N_time_steps
            )
        return time_steps, sensor_data

    data1 = generate_data(101)
    data2 = generate_data(51)

    me1 = MyModelError(my_forward_model, data1)
    me2 = MyModelError(my_forward_model, data2)

    problem = InferenceProblem()
    problem.add_model_error(me1)
    problem.add_model_error(me2)

    problem.latent["A"].add(me1, "A")
    problem.latent["A"].add(me2, "A")
    # or alternatively for "B"
    problem.latent["B"].add_shared()

    problem.latent["A"].prior = 40.0, 5.0
    problem.latent["B"].prior = 6000.0, 300.0

    for sensor in [s1, s2, s3]:
        noise_model = UncorrelatedSensorNoise([sensor])
        noise_key = problem.add_noise_model(noise_model)
        problem.latent_noise[noise_key].add(noise_model, "precision")
        problem.latent_noise[noise_key].prior = bayes.vb.Gamma.FromSD(noise_sds[sensor])
    
    info = VariationalBayesSolver(problem).estimate_parameters()
    print(info)

    """
    We now transform the vb problem into a sampling problem. 

    1)  Wrap problem.loglike for a tool of your choice
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
    2)  Define prior distributions in a tool of your choice!
    """
    import pymc3 as pm

    pymc3_prior = []

    model = pm.Model()
    with model:
        for name, latent in problem.latent.items():
            mean, sd = latent.prior
            pymc3_prior.append(pm.Normal(name, mu=mean, sigma=sd))

        for name, latent in problem.latent_noise.items():
            shape, scale = latent.prior.shape, latent.prior.scale
            alpha, beta = shape, 1.0 / scale
            pymc3_prior.append(pm.Gamma(name, alpha=alpha, beta=beta))

    """
    3)  Go!
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
    means = summary["mean"]

    for noise_key in problem.latent_noise:
        print(f"{noise_key}: vb    = ", 1.0 / info.noise[noise_key].mean ** 0.5)
        print(f"{noise_key}: pymc3 = ", 1.0 / means[noise_key] ** 0.5)

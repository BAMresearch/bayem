import unittest
import numpy as np

from bayes.parameters import ParameterList
from bayes.inference_problem import VariationalBayesProblem, ModelErrorInterface
from bayes.noise import UncorrelatedNoiseTerm

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

    problem = VariationalBayesProblem()
    key1 = problem.add_model_error(me1)
    key2 = problem.add_model_error(me2)

    problem.latent["A"].add(me1.parameter_list, "A")
    problem.latent["A"].add(me2.parameter_list, "A")
    # or simply
    problem.define_shared_latent_parameter_by_name("B")

    problem.set_normal_prior("A", 40.0, 5.0)
    problem.set_normal_prior("B", 6000.0, 300.0)

    noise1 = UncorrelatedNoiseTerm()
    noise1.add(s1, key1)
    noise1.add(s2, key1)
    noise1.add(s3, key1)

    noise2 = UncorrelatedNoiseTerm()
    noise2.add(s1, key2)
    noise2.add(s2, key2)
    noise2.add(s3, key2)

    noise_key1 = problem.add_noise_model(noise1)
    noise_key2 = problem.add_noise_model(noise2)

    problem.set_noise_prior(noise_key1, 3 * noise_sd1, sd_shape=0.5)
    problem.set_noise_prior(noise_key2, 3 * noise_sd2, sd_shape=0.5)

    info = problem.run()
    print(info)

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

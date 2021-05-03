import unittest
import numpy as np

from bayes.parameters import ParameterList
# from bayes.inference_problem import VariationalBayesProblem, ModelErrorInterface

import torch as th

import pyro
from pyro.distributions import Normal, Uniform, MultivariateNormal
from pyro.infer import EmpiricalMarginal, Importance, NUTS, MCMC

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
        # return f"{self.name} {self.shape}"
        return f"{self.name}"


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
        # result=th.reshape(th.tensor(list(result.values())),(-1,))
        # result=list(result.values())
        return result

    def parameter_list(self):
        p = ParameterList()
        p.define("A", None)
        p.define("B", None)
        return p


# ----------------------------------------
if __name__ == "__main__":
    s1, s2, s3 = MySensor("S1", 0.2), MySensor("S2", 0.5), MySensor("S3", 42.0)

    fw = MyForwardModel()
    prm = fw.parameter_list()

    # set the correct values
    A_correct = 42.0
    B_correct = 6174.0
    prm["A"] = A_correct
    prm["B"] = B_correct

    np.random.seed(6174)
    noise_sd1 = 10


    def generate_data(N_time_steps, noise_sd):
        time_steps = np.linspace(0, 1, N_time_steps)
        model_response = fw(prm, [s1, s2, s3], time_steps)
        sensor_data = {}
        for sensor, perfect_data in model_response.items():
            sensor_data[sensor] = perfect_data + np.random.normal(
                0.0, noise_sd, N_time_steps
            )
        #    return time_steps, sensor_data
        return sensor_data


    data1 = generate_data(20, noise_sd1)
    data_obs_array = th.tensor(np.reshape(list(data1.values()), (-1,)))


    # ---------------------------------------------------
    # A is unknown input and B,t,x is known input.

    def model(observed_data):
        A_prior = Normal(40, 100)  ## keeping it heavily uninformed
        A = pyro.sample("A", A_prior)

        def observe_A_lkl(observed_data, obs_name):
            fw = MyForwardModel()
            prm = fw.parameter_list()
            prm["A"] = A
            prm["B"] = th.tensor(B_correct)
            N_time_steps = 20
            time_steps = np.linspace(0, 1, N_time_steps)
            model_response = fw(prm, [s1, s2, s3], time_steps)
            model_res_array = []
            for sensor, mod_r in model_response.items():
                model_res_array.append(mod_r)

            model_res_array = th.cat((model_res_array))

            likelihood = MultivariateNormal(model_res_array.float(),
                                            covariance_matrix=th.tensor(noise_sd1) * th.eye(
                                                np.size(observed_data.numpy())))
            pyro.sample(obs_name, likelihood, obs=observed_data)

        observe_A_lkl(observed_data, "obs_1")

        return A


    def main_MCMC(n):
        # data=th.randn(500, 1)
        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_samples=n)
        mcmc.run(data_obs_array.float())
        posterior = mcmc.get_samples()['A'].numpy()

        posterior_mean = np.mean(posterior)
        posterior_std_dev = np.std(posterior)

        # report results
        inferred_mu = posterior_mean
        inferred_mu_uncertainty = 2*posterior_std_dev
        # print("The true Youngs modulus was %.3f " % E)
        print("The unknown parameter A inferred is %.3f +- %.4f" %
              (inferred_mu, inferred_mu_uncertainty))


    main_MCMC(100)
    print('The A_correct was taken as %.3f' % A_correct)

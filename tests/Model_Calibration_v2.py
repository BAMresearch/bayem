# ---------
# AUTHOR: Atul Agrawal (atul.agrawal@tum.de)
# ---------

import unittest
import numpy as np

from bayes.parameters import ParameterList
from bayes.inference_problem import VariationalBayesProblem, ModelErrorInterface
from bayes.Caliberation import Inference

import torch as th


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


# --- Wrapper for the forward solve class
def wrapper_forward(B_correct, latent_para):
    """
    Takes in unkown and known inputs and returns the function output as vector
    :param B_correct:
    :param latent_para:
    :return:
    """
    fw = MyForwardModel()
    prm = fw.parameter_list()
    prm["A"] = latent_para
    prm["B"] = th.tensor(B_correct)

    N_time_steps = 40
    time_steps = np.linspace(0, 1, N_time_steps)
    model_response = fw(prm, [s1, s2, s3], time_steps)
    model_res_array = []
    for sensor, mod_r in model_response.items():
        model_res_array.append(mod_r)

    model_res_array = th.cat((model_res_array))
    return model_res_array.float()


if __name__ == "__main__":
    # ------Do experiment (Generate noisy synthetic data)
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


    data1 = generate_data(40, noise_sd1)
    data_obs_array = th.tensor(np.reshape(list(data1.values()), (-1,))).float()

    # -------Metadata for Inference problem
    # TODO: Pass this through some JSON/YAML file
    prior_hyperparameter = [40, 100]
    prior_dist = "Normal"
    forward_solve_wrapper = wrapper_forward
    forward_solve_known_input = B_correct
    Observed_data = data_obs_array
    Noise_distribution = "Normal"
    Noise_hyperparameter = noise_sd1

    # -------- Setup the Inference problem
    infer = Inference(prior_dist, prior_hyperparameter, forward_solve_wrapper, forward_solve_known_input, Observed_data,
                      Noise_distribution, Noise_hyperparameter)

    # --------- Solve the Inference problem
    posterior = infer.run(1000, kernel="NUTS")

    # --------- Visualise
    infer.visualize(posterior)

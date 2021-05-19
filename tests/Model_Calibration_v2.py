# ---------
# AUTHOR: Atul Agrawal (atul.agrawal@tum.de)
# ---------

import unittest
import numpy as np

from bayes.parameters import ParameterList
from bayes.inference_problem import VariationalBayesProblem, ModelErrorInterface
from bayes.Calibration import Inference

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
def wrapper_forward(known_input, latent_para):
    """
    Takes in unkown (latent) and known inputs and returns the function output as vector
    :param known_input: (Dict type) ('known_parameters': , 'sensors': ,'time_steps': )
    :param latent_para:
    :return:
    """
    known_para = known_input['known_parameters']
    sensor_pos = known_input['sensors']
    tot_time_steps = known_input['time_steps']

    fw = MyForwardModel()
    prm = fw.parameter_list()
    prm["A"] = latent_para
    prm["B"] = th.tensor(known_para)

    time_steps = np.linspace(0, 1, tot_time_steps)
    #model_response = fw(prm, [s1, s2, s3], time_steps)
    model_response = fw(prm, sensor_pos, time_steps)
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

    N_time_steps = 40
    data1 = generate_data(N_time_steps, noise_sd1)
    data_obs_array = th.tensor(np.reshape(list(data1.values()), (-1,))).float()

    # ----------------------------------------------
    # -------------- Inference and prediction part starts from here
    # -------------------------------------------------

    # -- Metadata for Inference problem
    # TODO: Pass this through some JSON/YAML file
    prior_hyperparameter = [40, 10]
    prior_dist = "Normal"
    Observed_data = data_obs_array
    Noise_distribution = "Normal"
    Noise_hyperparameter = noise_sd1 #TODO: test with correlated noise model

    # ---- Metadata for forward solve
    forward_solve_wrapper = wrapper_forward
    forward_solve_known_para = B_correct
    sensors = [s1,s2,s3]
    N_time_steps = N_time_steps
    forward_input = {'known_parameters': B_correct, 'sensors': sensors, 'time_steps': N_time_steps}

    # -- Setup the Inference problem
    infer = Inference(prior_dist, prior_hyperparameter, forward_solve_wrapper, forward_input, Observed_data,
                      Noise_distribution, Noise_hyperparameter)

    # -- Solve the Inference problem
    # ---------- learns the posterior of latent parameter just from the noisy observed data
    posterior = infer.run(1000, kernel="NUTS")

    # -- Visualise
    infer.visualize_prior_posterior(posterior)

    # -- Predict
    new_input_forward = {'known_parameters': B_correct, 'sensors': [s1,s2], 'time_steps': 2}
    pred_pos = infer.predict(posterior,new_input_forward)
    print("The predicted value of the forward solve is:")

    infer.visualize_predictive_posterior(pred_pos) # visualise the posterior predictive

    ground_truth = wrapper_forward(new_input_forward,A_correct)

    print("The ground truth of the forward solve is:")
    print(ground_truth)
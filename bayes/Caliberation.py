
import numpy as np
import torch as th

import pyro
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal, Importance ,NUTS ,MCMC

import seaborn as sns
import matplotlib.pyplot as plt

class Inference:
    def __init__(self, prior_dist, prior_hyperparameters, forward_solve, fw_input, observed_data, obs_noise_dist,
                 obs_noise_parameters):
        """

        :param prior_dist:
        :param prior_hyperparameters:
        :param forward_solve:
            * A wrapper to the forward solve to be provided here which internally calls the forward solve,
            takes in known input given by fw_input and takes in parameters to be calibrated and returns the model output which acts as a mean.

        :param fw_input:
        :param observed_data:
        :param obs_noise_dist:
        :param obs_noise_parameters:
        """
        self.prior_dist = prior_dist
        self.prior_hyperparameters = prior_hyperparameters
        self.forward_solve = forward_solve
        self.fw_input = fw_input
        self.observed_data = observed_data
        self.obs_noise_dist = obs_noise_dist
        self.obs_noise_parameters = obs_noise_parameters

    def model(self, observed_data):
        # --prior
        #TODO: Incorporate prior_dist user choice here
        self.para_prior = dist.Normal(self.prior_hyperparameters[0], self.prior_hyperparameters[1])
        para = pyro.sample("theta", self.para_prior)

        # --likelihood
        mean = self.forward_solve(self.fw_input, para)
        #TODO: Incorporate noise model dist choice here, default is normal
        self.likelihood = dist.Normal(mean, self.obs_noise_parameters ** 2)
        pyro.sample("lkl", self.likelihood, obs=observed_data)

        return para

    def run(self, n, kernel=None):

        if kernel is None:
            kernel = NUTS(self.model)
        if kernel == "HMC":
            print('Using HMC kernel')
            kernel = HMC(model=self.model, step_size=1, num_steps=4)

        mcmc = MCMC(kernel, num_samples=n)
        mcmc.run(self.observed_data)
        posterior = mcmc.get_samples()['theta'].numpy()
        # print(posterior)
        mcmc.summary()

    def visualize():
        raise NotImplementedError

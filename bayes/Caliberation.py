#---------
# AUTHOR: Atul Agrawal (atul.agrawal@tum.de)
#---------
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

        :param prior_dist: [string] Specify the prior distribution
        :param prior_hyperparameters: [2x1] Specify the hyperparameters of the prior dist.
        :param forward_solve:
            * A wrapper to the forward solve to be provided here which internally calls the forward solve,
            takes in known input given by fw_input and takes in parameters to be calibrated and returns the model output which acts as a mean.

        :param fw_input: [] known input to the forward solve
        :param observed_data [N,] [tensor]: The observed noisy data
        :param obs_noise_dist: [string] - Specify the noise distribution
        :param obs_noise_parameters: Hyperaparameters of the noise model
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
        #--- https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
        #TODO: Incorporate prior_dist user choice here
        if self.prior_dist == "Normal":
            self.para_prior = dist.Normal(self.prior_hyperparameters[0], self.prior_hyperparameters[1])
        if self.prior_dist == "Uniform":
            self.para_prior = dist.Uniform(self.prior_hyperparameters[0], self.prior_hyperparameters[1])
        if self.prior_dist == "Gamma":
            self.para_prior = dist.Gamma(self.prior_hyperparameters[0], self.prior_hyperparameters[1])
        if self.prior_dist == "Beta":
            self.para_prior = dist.Beta(self.prior_hyperparameters[0], self.prior_hyperparameters[1])
        para = pyro.sample("theta", self.para_prior)
        # --likelihood
        mean = self.forward_solve(self.fw_input, para)
        #TODO: Incorporate noise model dist choice here, default is normal
        if self.obs_noise_dist == "Normal":
            self.likelihood = dist.Normal(mean, self.obs_noise_parameters ** 2)
        else:
            raise NotImplementedError
        pyro.sample("lkl", self.likelihood, obs=observed_data)

        return para

    def run(self, n, kernel=None):
        #TODO: Incorporate VI
        if kernel == "NUTS":
            kernel = NUTS(self.model)
        if kernel == "HMC":
            print('Using HMC kernel')
            kernel = HMC(model=self.model, step_size=1, num_steps=4)

        mcmc = MCMC(kernel, num_samples=n)
        mcmc.run(self.observed_data)
        posterior = mcmc.get_samples()['theta'].numpy()
        # print(posterior)
        mcmc.summary()
        return posterior

    def visualize(self, posterior):
        """

        :param posterior: Pass the posterior after MCMC
        :return:
        """

        plt.figure(figsize=(3, 3))
        sns.histplot(posterior, kde=True, label="para_posterior", bins=20)
        plt.legend()
        plt.xlabel("parameter")
        plt.ylabel("Samples")
        plt.show()

        # plotting priors
        smpl = np.ndarray((100000))
        for i in range(1, 100000):
            smpl[i] = (pyro.sample("AA", self.para_prior))
        plt.figure(figsize=(3, 3))
        sns.kdeplot(data=smpl, label="para_prior")
        plt.legend()
        plt.xlabel("parameter")
        plt.ylabel("Density")
        plt.show()

        #raise NotImplementedError

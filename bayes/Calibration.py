# ---------
# AUTHOR: Atul Agrawal (atul.agrawal@tum.de)
# ---------
import numpy as np
import torch as th

import torch as th
import pyro
import pyro.distributions as dist
from numpy.core._multiarray_umath import ndarray
from pyro.infer import EmpiricalMarginal, Importance, NUTS, MCMC, HMC
import seaborn as sns
import matplotlib.pyplot as plt

from utils.forwardSolverInterface import forwardSolverInterface


class Inference:
    def __init__(self, prior_dist, prior_hyperparameters, forward_solve, fw_input, observed_data, obs_noise_dist,
                 obs_noise_parameters, hyperprior_dist, hyperprior_para):
        """

        :param prior_dist: [string] Specify the prior distribution
        :param prior_hyperparameters: [2x1] Specify the hyperparameters of the prior dist.
        :param forward_solve: A called wrapper class, with input and outputs as [tensor]
            * A wrapper to the forward solve to be provided here which internally calls the forward solve,
            takes in known input given by fw_input and takes in parameters to be calibrated and returns the model output.

        :param fw_input: [Dict type] ('known_parameters': , 'sensors': ,'time_steps': )
        :param observed_data [N,] [tensor]: The observed noisy data
        :param obs_noise_dist: [string] - Specify the noise distribution
        :param obs_noise_parameters: Hyperaparameters of the noise model (Can be None when it is to be inferred)
        """

        self.prior_dist = prior_dist
        self.prior_hyperparameters = prior_hyperparameters
        self.forward_solve = forward_solve
        self.fw_input = fw_input
        self.observed_data = observed_data
        self.obs_noise_dist = obs_noise_dist
        self.obs_noise_parameters = obs_noise_parameters
        self.hyperprior_dist = hyperprior_dist
        self.hyperprior_para = hyperprior_para

    def posterior_model(self, observed_data):
        """
        Model to construct prior, likelihood and the posterior with the supplied Observed data
        :param observed_data:
        :return:
        #TODO: Infer likelihood noise/hyperprior (atleast the MAP)
        """
        # --prior
        # --- https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
        # TODO: Incorporate prior_dist user choice here
        if self.prior_dist == "Normal":
            self.para_prior = dist.Normal(self.prior_hyperparameters[0], self.prior_hyperparameters[1])
        if self.prior_dist == "Uniform":
            self.para_prior = dist.Uniform(self.prior_hyperparameters[0], self.prior_hyperparameters[1])
        if self.prior_dist == "Gamma":
            self.para_prior = dist.Gamma(self.prior_hyperparameters[0], self.prior_hyperparameters[1])
        if self.prior_dist == "Beta":
            self.para_prior = dist.Beta(self.prior_hyperparameters[0], self.prior_hyperparameters[1])
        para = pyro.sample("theta", self.para_prior)

        # --hyperprior
        if self.obs_noise_parameters is None:
            # TODO: To add hyperparamters for the hyperprior
            # sigma_prior = dist.Normal(0, 1) # AA: Hardcoded mean close to the known noise
            if self.hyperprior_dist == "Gamma":
                sigma_prior = dist.Gamma(self.hyperprior_para[0], self.hyperprior_para[1])
            sigma_noise = pyro.sample("sigma", sigma_prior)
            # self.obs_noise_parameters = sigma_noise
        if self.obs_noise_parameters is not None:
            sigma_noise = self.obs_noise_parameters

        # --likelihood
        # ---- wrap forward solver in a forward solver interface with overriden autograd
        BB_solver = False
        if BB_solver:
            forward = forwardSolverInterface(self.forward_solve, self.fw_input)
            forward_autograd = forward.override_autograd()
            mean = forward_autograd(para)
        mean = self.forward_solve(self.fw_input, para)
        # TODO: Incorporate noise model dist choice here, default is normal
        # TODO: More involved noise model, with correlation structure
        if self.obs_noise_dist == "Normal":
            self.likelihood = dist.Normal(mean, sigma_noise)
        else:
            raise NotImplementedError
        pyro.sample("lkl", self.likelihood, obs=observed_data)

        return para

    def run(self, n, kernel=None):
        """
        Method to find approximate posterior with MCMC/VI
        :param n:
        :param kernel:
        :return: posterior: [N,]
        """
        # TODO: Incorporate VI
        if kernel == "NUTS":
            kernel = NUTS(self.posterior_model)
        if kernel == "HMC":
            print('Using HMC kernel')
            kernel = HMC(model=self.posterior_model, step_size=1, num_steps=4)

        mcmc = MCMC(kernel, num_samples=n)
        mcmc.run(self.observed_data)
        posterior_para = mcmc.get_samples()['theta'].numpy()
        posterior_noise = mcmc.get_samples()['sigma'].numpy()
        mcmc.summary()
        return posterior_para, posterior_noise

    def predict(self, posterior_para, posterior_noise, new_input):
        """
        Method to get posterior predictive distribution. Integration approximated using Monte Carlo.
        Current assumption is no model Bias, just observational noise
        :param posterior_para: p(theta|D) samples
        :param posterior_noise: p(sigma|D) posterior of the observational noise
        :param new_input: New input to the solver. [Dict type] ('known_parameters': , 'sensors': ,'time_steps': )
        :return: tilda_X [S x T]: New unobserved data samples., with S being number of sample and T being the parameters
        """
        # size = np.size(new_input['sensors']) * new_input['time_steps']
        size = [np.size(v) for v in new_input.values()]
        # size = np.size(new_input['known_inputs'])
        tilda_X = np.ndarray((np.size(posterior_para), size[0]))
        sigma_mean = np.max(posterior_noise)  # AA : Just using MAP point for the sigma posterior, more involved
        # would be an inner loop for sigma also.
        for i in range(0, np.size(posterior_para)):
            theta = posterior_para[i]
            mean = self.forward_solve(new_input, theta)
            # _dist = dist.Normal(mean, sigma_mean)
            # tilda_X[i, :] = pyro.sample("pos", _dist)
            tilda_X[i, :] = mean
        return tilda_X

    def visualize_prior_posterior(self, posterior_para, posterior_noise):
        """

        :param posterior: Pass the posterior after MCMC
        :return:
        """

        plt.figure(figsize=(3, 3))
        # sns.histplot(posterior, kde=True, label="para_posterior", bins=20)
        sns.kdeplot(data=posterior_para, label="para_posterior")
        # plotting priors
        smpl = np.ndarray((100000))
        for i in range(1, 100000):
            smpl[i] = (pyro.sample("AA", self.para_prior))
        # plt.figure(figsize=(3, 3))
        sns.kdeplot(data=smpl, label="para_prior",
                    clip=(np.mean(posterior_para) - 3 * np.std(posterior_para),
                          np.mean(posterior_para) + 3 * np.std(posterior_para)))
        plt.legend()
        plt.show()

        plt.figure(figsize=(3, 3))
        sns.kdeplot(data=posterior_noise, label="s.d of noise_posterior")
        plt.legend()

        plt.show()

        # raise NotImplementedError

    def visualize_predictive_posterior(self, pred_posterior):
        """

        :param pred_posterior: [NxM] Samples of the predictive posterior, with N being the total number of samples
        and M being the number of experiments. :return:
        """
        for i in range(0, np.shape(pred_posterior)[1]):
            plt.figure(figsize=(3, 3))
            sns.kdeplot(data=pred_posterior[:, i], label="predictive_posterior")
            plt.legend()
            plt.xlabel("Y")
            plt.show()

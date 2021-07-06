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
import pandas as pd

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
        :return: posterior: Dict of the all the parameters
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
        posterior = {'parameter': posterior_para, 'noise_sd': posterior_noise}
        return posterior

    def predict(self, posterior, new_input,model_discrepancy=False):
        """
        Method to get posterior predictive distribution. Integration approximated using Monte Carlo.
        Current assumption is no model Bias, just observational noise
        :param posterior: Dict of the all the parameters with keys 'parameter' and 'noise_sd'
        :param new_input: New input to the solver. [Dict type] ('known_parameters': , 'sensors': ,'time_steps': )
        :param model_discrepancy: If its True, then additive noise is due to model discrepancy, else due to sensor noise.
                In the materials domain, as reported by BAM people, model discrepancy is much more pronounced than sensor noise.
        :return: tilda_X [S x T]: New unobserved data samples., with S being number of sample and T being the parameters
        """
        posterior_para = posterior['parameter']
        posterior_noise = posterior['noise_sd']

        size = [np.size(v) for v in new_input.values()]
        # size = np.size(new_input['known_inputs'])
        tilda_X = np.ndarray((np.size(posterior_para), size[0]))
        sigma_mean = np.max(posterior_noise)  # AA : Just using MAP point for the sigma posterior, more involved
        # would be an inner loop for sigma also.
        for i in range(0, np.size(posterior_para)):
            theta = posterior_para[i]
            mean = self.forward_solve(new_input, theta)
            _dist = dist.Normal(mean, sigma_mean)
            if model_discrepancy:
                tilda_X[i, :] = pyro.sample("pos", _dist)
            else:
                tilda_X[i, :] = mean
        return tilda_X

    def visualize_prior_posterior(self, posterior, pairplot = False):
        """
        Returns KDE plots of the posterior, plus optionally pair plots.
        :param posterior: Dict of the all the parameters with keys 'parameter' and 'noise_sd'
        :return:
        """
        posterior_para = posterior['parameter']
        posterior_noise = posterior['noise_sd']

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
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        sns.kdeplot(data=posterior_noise, label="s.d of noise_posterior")
        plt.legend()

        plt.show()

        if pairplot:
            sns.pairplot(pd.DataFrame(data=posterior), kind="kde")


    def visualize_predictive_posterior(self, pred_posterior, test_data):
        """

        :param pred_posterior: [S x T] New unobserved data samples., with S being number of sample and T being the parameters
        :param test_data: [Dict] Contains stress and strain experimental values to compare our predictions
        :return:
        """
        pos = np.quantile(pred_posterior, [0.05, 0.5, 0.95], axis=0)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.fill_betweenx(test_data['stress'], pos[0, :], pos[2, :], alpha=0.4, label='Predicted strain')
        plt.plot(test_data['strain'], test_data['stress'], 'g', label='Experimental strain')
        plt.legend()
        plt.xlabel('strain')
        plt.ylabel('stress')

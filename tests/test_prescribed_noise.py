#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 2021

@author: ajafari
"""

import unittest
import numpy as np
from bayes import vb
from taralli.parameter_estimation.base import *
import scipy.stats


class ME_LinearRegression:
    def __init__(self, Xs, Ys):
        self.Xs = np.array(Xs)
        self.Ys = np.array(Ys)
        self.n_evals = 0
    
    def __call__(self, X):
        a = X[0]
        b = X[1] # offset
        residuals = (a * self.Xs + b) - self.Ys
        self.n_evals += 1
        return {'res': residuals}


def taralli_solve(me, prior_pars, noise0, infer_noise):
    priors = []
    for i in range(len(prior_pars.mean)):
        priors.append(scipy.stats.norm(loc=prior_pars.mean[i], scale=prior_pars.std_diag[i]))

    if infer_noise:
        priors.append(scipy.stats.gamma(a=noise0["res"].shape, scale=noise0["res"].scale))


    def loglike(theta):
        errors = me(theta[:2])["res"]
        
        if infer_noise:
            if theta[2] < 0:
                return -np.inf
            prec = theta[2]
        else:
            prec = noise0["res"].shape * noise0["res"].scale

        sigma = 1 / prec**0.5
        return np.sum(scipy.stats.norm.logpdf(errors, scale=sigma))
    
    def ppf(theta):
        return np.array([prior.ppf(t) for prior, t in zip(priors, theta)])
    
    def logprior(theta):
        return sum([prior.logpdf(t) for prior, t in zip(priors, theta)])

    init = np.empty((20, len(priors)))
    for i, prior in enumerate(priors):
        init[:, i] = prior.rvs(20)

    nestle=True
    if nestle:
        model = NestleParameterEstimator(ndim=len(priors), prior_transform=ppf, log_likelihood=loglike, seed=6174)
        model.estimate_parameters(npoints=1000)
    else:
        model = EmceeParameterEstimator(nwalkers=20, ndim=len(priors), log_prior=logprior, log_likelihood=loglike, seed=6174, sampling_initial_positions=init)
        model.estimate_parameters()
    model.summary()
    return model.summary_output["mean"], model.summary_output["covariance"]


class TestPrescribedNoise(unittest.TestCase):
    def setUp(self):
        n_data = 100
        L = 10
        Xs = np.linspace(0, L, n_data)
        ## Target values
        A = 129.0
        B = -1890.0
        ## Perfect data Ys
        Ys_perfect = A * Xs + B
        ## Noisy data
        Ys_max = max(abs(Ys_perfect))
        np.random.seed(13)
        std_noise = Ys_max * 0.02
        noise = np.random.normal(0., std_noise, n_data)
        Ys = Ys_perfect + noise # noisy data
        ## MODEL
        self.model = ME_LinearRegression(Xs, Ys)
        ## PRIOR parameters
        means0 = [ A,  B]
        stds0 = [0.01*A,  0.01*B]
        precisions0 = [1.0 / (s**2) for s in stds0]
        self.prior_pars = vb.MVN(means0, np.diag(precisions0))
        ## PRIOR noise
        self.noise0 = dict()
        sh0, sc0 = 1, std_noise # based on std of Ys
        self.noise0['res'] = vb.Gamma(shape=sh0, scale=sc0)
    
    def test_with_noise(self, tolerance=1e-4, print_=True):
        ## VB solution
        update_noise = {'res': True}
        vb_outputs = vb.variational_bayes(self.model, self.prior_pars, tolerance=tolerance, noise0=self.noise0, update_noise=update_noise)    
        if print_:
            print(vb_outputs)
        ## TARALLI (Sampling) solution
        means, cov = taralli_solve(self.model, self.prior_pars, self.noise0, infer_noise=True)
        ## COMPARIOSINs
        mean_pars = means[:2]
        cov_pars = cov[:2, :2]
        prec_noise = means[2]
        np.testing.assert_almost_equal(vb_outputs.param.mean / mean_pars, np.ones(2), decimal=3)
        np.testing.assert_almost_equal(0.5 * vb_outputs.param.cov / cov_pars, 0.5 * np.ones((2,2)), decimal=2)
        np.testing.assert_almost_equal(vb_outputs.noise['res'].mean / prec_noise, 1, decimal=2) # compare noise precision (mean value)

    
    def test_without_noise(self, tolerance=1e-4, print_=True):
        ## VB solution
        update_noise = {'res': False}
        vb_outputs = vb.variational_bayes(self.model, self.prior_pars, tolerance=tolerance, noise0=self.noise0, update_noise=update_noise)    
        if print_:
            print(vb_outputs)
        ## TARALLI (Sampling) solution
        means, cov = taralli_solve(self.model, self.prior_pars, self.noise0, infer_noise=False)
        ## COMPARIOSINs
        mean_pars = means[:2]
        cov_pars = cov[:2, :2]
        np.testing.assert_almost_equal(vb_outputs.param.mean / mean_pars, np.ones(2), decimal=3)
        np.testing.assert_almost_equal(vb_outputs.param.cov / cov_pars, np.ones((2,2)), decimal=2)

    
if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 11:39:42 2021

@author: ajafari

Abbreviations:
    "me": model error
    "fw": forward model
    "us": related to displacements
    "fs": related to forces
    "sd" and "std": standard deviation
    "prec": precision

The goal is to investigate how sensitive the VB is with respect to:
    - parameterization of input latent parameters (here, "A" and "B")
    - any partial weightings in the output of the model (here, "weight_fw_fs")
"""
import unittest
import numpy as np
from bayes.vb import *
import matplotlib.pyplot as plt

scale_fs = 100.0 # some scale to make forces larger than displacements
us = np.linspace(0, 1, 1000) # for displacements
fs = np.linspace(0, 1, 1000) # for forces
A, B = 7.0, 42.0
sd_us, sd_fs = 0.02, 0.01 * scale_fs
prior_factors = [1.3, 0.5] # are multiplied with target values of parameters to give us prior means
prior_factors_sds = [0.3, 0.4] # are multiplied with target values of parameters to give us prior standard deviations

def fw_us(ps):
    # return ps[0] * us + ps[1]
    return np.log(abs(ps[0] * us + np.sqrt(abs(ps[1] * us)) +1) )
def fw_fs(ps):
    # return scale_fs * (ps[0] * fs + ps[1])
    return scale_fs * ( np.sqrt(abs(ps[0] * fs)) + np.log(abs(ps[1] * fs) + 1) )

class LinearActualizer:
    def __init__(self, ref):
        self.ref = ref
    def __call__(self, x):
        return self.ref * x

class ExponentialActualizer:
    def __init__(self, ref):
        self.ref = ref
    def __call__(self, x):
        return self.ref * np.exp(x)

def proper_gammas(weight_fw_fs):
    target_prec_us = 1.0 / sd_us**2
    target_prec_fs = 1.0 / (weight_fw_fs * sd_fs)**2
    g0_us = Gamma(0.1, target_prec_us/0.1) # Note that the mean is set to target, but the std is quite large.
    g0_fs = Gamma(0.1, target_prec_fs/0.1)
    return g0_us, g0_fs, [target_prec_us, target_prec_fs]

def print_all(info, target_pars, target_precs, param_prior, noise_prior):
    print('### Target parameters ###'); print(target_pars)
    print('### Target noises (precision) ###'); print(target_precs)
    print('### Prior parameters ###'); print(param_prior)
    print('### Prior noises (precision) ###'); print(noise_prior)
    print(info)

class Me:
    def __init__(self, weight_fw_fs):
        self.weight_fw_fs = weight_fw_fs
        self._set_data()
    def __call__(self, parameters):
        ps = []
        for i, a in enumerate(self.actualizers):
            ps.append(a(parameters[i]))
        er = {"noise_us": fw_us(ps) - self.data_us}
        er.update({"noise_fs": self.weight_fw_fs * fw_fs(ps) - self.data_fs})
        return er
    def set_actualizers(self, actualizers):
        assert len(actualizers) == 2 # for A and B
        self.actualizers = actualizers
    def _set_data(self, _plot=True):
        np.random.seed(6174)
        self.data_us = fw_us([A, B]) + np.random.normal(0, sd_us, len(us))
        np.random.seed(1234)
        self.data_fs = self.weight_fw_fs * ( fw_fs([A, B]) + np.random.normal(0, sd_fs, len(fs)) )
        if _plot:
            plt.figure()
            plt.suptitle(f"Noisy data_us - weight_fs={self.weight_fw_fs}")
            plt.subplot(1,2,1)
            plt.plot(self.data_us)
            plt.subplot(1,2,2)
            plt.plot(self.data_fs)
            plt.show()

class Test_VB(unittest.TestCase):
    
    def _assertions(self, info, target_pars, target_precs):
        param_post, noise_post_dic = info.param, info.noise
        for i, correct_value in enumerate(target_pars):
            posterior_mean = param_post.mean[i]
            posterior_std = param_post.std_diag[i]

            self.assertLess(posterior_std, 0.3)
            self.assertAlmostEqual(
                posterior_mean, correct_value, delta=2 * posterior_std
            )
        for i, noise_post in enumerate(noise_post_dic.values()):
            prec = target_precs[i]
            post_noise_precision = noise_post.mean
            self.assertAlmostEqual(post_noise_precision, prec, delta=0.1 * prec)
            self.assertLess(info.nit, 20)
    
    def test_vb_linear_weight(self):
        weight_fw_fs = 1.0 / scale_fs
        me = Me(weight_fw_fs)
        actualizers = [LinearActualizer(A), LinearActualizer(B)]
        me.set_actualizers(actualizers)
        target_pars = [1, 1] # due to linear actualizers with ref=target_parameter
        param_prior = MVN(prior_factors, [[1 / prior_factors_sds[0] ** 2, 0], [0, 1 / prior_factors_sds[1] ** 2]])
        
        g0_us, g0_fs, target_precs = proper_gammas(weight_fw_fs)
        noise_prior = {'noise_us': g0_us, 'noise_fs': g0_fs}

        info = variational_bayes(me, param_prior, noise_prior)
        print('--------------- Linear parameterization - Weight in output')
        print_all(info, target_pars, target_precs, param_prior, noise_prior)
        # self._assertions(info, target_pars, target_precs)
    
    def test_vb_linear_noweight(self):
        weight_fw_fs = 1.0
        me = Me(weight_fw_fs)
        actualizers = [LinearActualizer(A), LinearActualizer(B)]
        me.set_actualizers(actualizers)
        target_pars = [1, 1] # due to linear actualizers with ref=target_parameter
        param_prior = MVN(prior_factors, [[1 / prior_factors_sds[0] ** 2, 0], [0, 1 / prior_factors_sds[1] ** 2]])
        
        g0_us, g0_fs, target_precs = proper_gammas(weight_fw_fs)
        noise_prior = {'noise_us': g0_us, 'noise_fs': g0_fs}

        info = variational_bayes(me, param_prior, noise_prior)
        print('--------------- Linear parameterization - NO weight in output')
        print_all(info, target_pars, target_precs, param_prior, noise_prior)
        # self._assertions(info, target_pars, target_precs)
    
    def test_vb_exp_weight(self):
        weight_fw_fs = 1.0 / scale_fs
        me = Me(weight_fw_fs)
        actualizers = [ExponentialActualizer(A), ExponentialActualizer(B)]
        me.set_actualizers(actualizers)
        target_pars = [0, 0] # due to exponential actualizers with ref=target_parameter
        stds = [i / j for i, j in zip(prior_factors_sds, prior_factors)]
        param_prior = MVN(np.log(prior_factors), [[1 / stds[0] ** 2, 0], [0, 1 / stds[1] ** 2]])
        
        g0_us, g0_fs, target_precs = proper_gammas(weight_fw_fs)
        noise_prior = {'noise_us': g0_us, 'noise_fs': g0_fs}

        info = variational_bayes(me, param_prior, noise_prior)
        print('--------------- Exponential parameterization - Weight in output')
        print_all(info, target_pars, target_precs, param_prior, noise_prior)
        # self._assertions(info, target_pars, target_precs)
    
    def test_vb_exp_noweight(self):
        weight_fw_fs = 1.0
        me = Me(weight_fw_fs)
        actualizers = [ExponentialActualizer(A), ExponentialActualizer(B)]
        me.set_actualizers(actualizers)
        target_pars = [0, 0] # due to exponential actualizers with ref=target_parameter
        stds = [i / j for i, j in zip(prior_factors_sds, prior_factors)]
        param_prior = MVN(np.log(prior_factors), [[1 / stds[0] ** 2, 0], [0, 1 / stds[1] ** 2]])
        
        g0_us, g0_fs, target_precs = proper_gammas(weight_fw_fs)
        noise_prior = {'noise_us': g0_us, 'noise_fs': g0_fs}

        info = variational_bayes(me, param_prior, noise_prior)
        print('--------------- Exponential parameterization - NO weight in output')
        print_all(info, target_pars, target_precs, param_prior, noise_prior)
        # self._assertions(info, target_pars, target_precs)

if __name__ == "__main__":
    unittest.main()
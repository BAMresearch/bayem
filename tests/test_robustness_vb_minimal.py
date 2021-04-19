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
sd_us, sd_fs = 0.03, 0.05 * scale_fs

def fw_us(ps):
    return ps[0] * us + ps[1]
    # return np.log(abs(ps[0] * us + 1)) + np.cos(ps[1])
def fw_fs(ps):
    return scale_fs * (ps[0] * fs + ps[1])
    # return scale_fs * (np.sin(ps[0] * fs * np.pi) + np.cos(ps[1]*2*np.pi))

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
    target_sd_us = 1.0 / sd_us**2
    target_sd_fs = 1.0 / (weight_fw_fs * sd_fs)**2
    g0_us = Gamma(0.1, target_sd_us/0.1)
    g0_fs = Gamma(0.1, target_sd_fs/0.1)
    return g0_us, g0_fs, [target_sd_us, target_sd_fs]

class Me:
    def __init__(self, weight_fw_fs):
        self.weight_fw_fs = weight_fw_fs
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
    def set_data(self, _plot=True):
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
    
    def _assertions(self, info, target_pars, target_sds):
        param_post, noise_post_dic = info.param, info.noise
        for i, correct_value in enumerate(target_pars):
            posterior_mean = param_post.mean[i]
            posterior_std = param_post.std_diag[i]

            self.assertLess(posterior_std, 0.3)
            self.assertAlmostEqual(
                posterior_mean, correct_value, delta=2 * posterior_std
            )
        for i, noise_post in enumerate(noise_post_dic.values()):
            sd = target_sds[i]
            post_noise_precision = noise_post.mean
            post_noise_std = 1.0 / post_noise_precision ** 0.5
            self.assertAlmostEqual(post_noise_std, sd, delta=0.03 * sd)
            self.assertLess(info.nit, 20)
    
    def test_vb_linear_weight(self):
        weight_fw_fs = 1.0 / scale_fs
        me = Me(weight_fw_fs)
        me.set_data()
        actualizers = [LinearActualizer(A), LinearActualizer(B)]
        me.set_actualizers(actualizers)
        target_pars = [1, 1]
        param_prior = MVN([1.3, 0.5], [[1 / 0.4 ** 2, 0], [0, 1 / 0.5 ** 2]])
        
        g0_us, g0_fs, target_sds = proper_gammas(weight_fw_fs)
        noise_prior = {'noise_us': g0_us, 'noise_fs': g0_fs}

        info = variational_bayes(me, param_prior, noise_prior)
        print('--------------- Linear parameterization - Weight in output')
        print(info)
        # self._assertions(info, target_pars, target_sds)
    
    def test_vb_exp_weight(self):
        weight_fw_fs = 1.0 / scale_fs
        me = Me(weight_fw_fs)
        me.set_data()
        actualizers = [ExponentialActualizer(A), ExponentialActualizer(B)]
        me.set_actualizers(actualizers)
        target_pars = [0, 0]
        param_prior = MVN([0.3, -0.2], [[1 / 0.2 ** 2, 0], [0, 1 / 0.3 ** 2]])
        
        g0_us, g0_fs, target_sds = proper_gammas(weight_fw_fs)
        noise_prior = {'noise_us': g0_us, 'noise_fs': g0_fs}

        info = variational_bayes(me, param_prior, noise_prior)
        print('--------------- Exponential parameterization - Weight in output')
        print(info)
        # self._assertions(info, target_pars, target_sds)
        
    def test_vb_linear_noweight(self):
        weight_fw_fs = 1.0
        me = Me(weight_fw_fs)
        me.set_data()
        actualizers = [LinearActualizer(A), LinearActualizer(B)]
        me.set_actualizers(actualizers)
        target_pars = [1, 1]
        param_prior = MVN([1.3, 0.5], [[1 / 0.4 ** 2, 0], [0, 1 / 0.5 ** 2]])
        
        g0_us, g0_fs, target_sds = proper_gammas(weight_fw_fs)
        noise_prior = {'noise_us': g0_us, 'noise_fs': g0_fs}

        info = variational_bayes(me, param_prior, noise_prior)
        print('--------------- Linear parameterization - NO weight in output')
        print(info)
        # self._assertions(info, target_pars, target_sds)
    
    def test_vb_exp_noweight(self):
        weight_fw_fs = 1.0
        me = Me(weight_fw_fs)
        me.set_data()
        actualizers = [ExponentialActualizer(A), ExponentialActualizer(B)]
        me.set_actualizers(actualizers)
        target_pars = [0, 0]
        param_prior = MVN([0.3, -0.2], [[1 / 0.2 ** 2, 0], [0, 1 / 0.3 ** 2]])
        
        g0_us, g0_fs, target_sds = proper_gammas(weight_fw_fs)
        noise_prior = {'noise_us': g0_us, 'noise_fs': g0_fs}

        info = variational_bayes(me, param_prior, noise_prior)
        print('--------------- Exponential parameterization - NO weight in output')
        print(info)
        # self._assertions(info, target_pars, target_sds)

if __name__ == "__main__":
    unittest.main()
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

class FW_us:
    def __init__(self):
        pass
    def __call__(self, ps):
        # return ps[0] * us + ps[1]
        u = 0.1 * (ps[0] * ps[0] * us + ps[1] * us)
        return 1.0 / (1 + np.exp(-u))
    def jac(self, ps):
        f = self(ps)
        df_du = f*(1-f)
        du_dps = 0.1 * np.array([2 * ps[0] * us, us]).T
        return np.array([df_du * du_dps[:,0], df_du * du_dps[:,1]]).T
class FW_fs:
    def __init__(self):
        pass
    def __call__(self, ps):
        # return scale_fs * (ps[0] * fs + ps[1])
        u = 0.2 * (ps[0] * fs + ps[1] * fs)
        return 1.0 / (1 + np.exp(-u))
    def jac(self, ps):
        f = self(ps)
        df_du = f*(1-f)
        du_dps = 0.2 * np.array([fs, fs]).T
        return np.array([df_du * du_dps[:,0], df_du * du_dps[:,1]]).T

class LinearActualizer:
    def __init__(self, ref):
        self.ref = ref
    def __call__(self, x):
        return self.ref * x
    def diff(self, x):
        return self.ref
    
class ExponentialActualizer:
    def __init__(self, ref):
        self.ref = ref
    def __call__(self, x):
        return self.ref * np.exp(x)
    def diff(self, x):
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
    plt.plot(info.free_energies, marker='.')
    plt.title('Free energies')
    plt.show()

def test_jacobian_fw(tol=1e-8):
    pars = [A, B]
    
    fw_us = FW_us()
    jac_fw_us = fw_us.jac(pars)
    
    fw_fs = FW_fs()
    jac_fw_fs = fw_fs.jac(pars)
    
    ## CD
    import copy
    jac_us_cd = []
    jac_fs_cd = []
    for i, p in enumerate(pars):
        dp = p * 2e-6
        pars_minus = copy.deepcopy(pars)
        pars_minus[i] = p - dp / 2
        pars_plus = copy.deepcopy(pars)
        pars_plus[i] = p + dp / 2
        
        fw_us_minus = fw_us(pars_minus)
        fw_us_plus = fw_us(pars_plus)
        fw_fs_minus = fw_fs(pars_minus)
        fw_fs_plus = fw_fs(pars_plus)
        
        jac_us_cd.append((fw_us_plus - fw_us_minus) / dp)
        jac_fs_cd.append((fw_fs_plus - fw_fs_minus) / dp)
    jac_us_cd = np.array(jac_us_cd).T
    jac_fs_cd = np.array(jac_fs_cd).T
    e_us = np.linalg.norm(jac_fw_us - jac_us_cd) / np.linalg.norm(jac_us_cd)
    e_fs = np.linalg.norm(jac_fw_fs - jac_fs_cd) / np.linalg.norm(jac_fs_cd)
    print(f"The relative error in Jacobian_us of forward_model is {e_us}")
    print(f"The relative error in Jacobian_fs of forward_model is {e_fs}")
    assert e_us<tol and e_fs<tol

def test_jacobian_me(tol = 1e-8):
    weight_fw_fs = 1.0
    me = Me(weight_fw_fs, False)
    actualizers = [LinearActualizer(A), LinearActualizer(B)]
    me.set_actualizers(actualizers)
    
    pars = [1.0, 1.0]
    jac_me = me.jacobian(pars)
    
    ## CD
    import copy
    jac_cd = {'noise_us': [], 'noise_fs': []}
    for i, p in enumerate(pars):
        dp = p * 2e-6
        pars_minus = copy.deepcopy(pars)
        pars_minus[i] = p - dp / 2
        pars_plus = copy.deepcopy(pars)
        pars_plus[i] = p + dp / 2
        
        me_minus = me(pars_minus)
        me_plus = me(pars_plus)
        
        jac_cd['noise_us'].append(-(me_plus['noise_us'] - me_minus['noise_us']) / dp) # "minus" is required due to an artifitial "_fact=-1" in me.jacobian
        jac_cd['noise_fs'].append(-(me_plus['noise_fs'] - me_minus['noise_fs']) / dp)
    jac_cd['noise_us'] = np.array(jac_cd['noise_us']).T
    jac_cd['noise_fs'] = np.array(jac_cd['noise_fs']).T
    e_us = np.linalg.norm(jac_me['noise_us'] - jac_cd['noise_us']) / np.linalg.norm(jac_cd['noise_us'])
    e_fs = np.linalg.norm(jac_me['noise_fs'] - jac_cd['noise_fs']) / np.linalg.norm(jac_cd['noise_fs'])
    print(f"The relative error in Jacobian_us of model_error is {e_us}")
    print(f"The relative error in Jacobian_fs of model_error is {e_fs}")
    assert e_us<tol and e_fs<tol

class Me:
    def __init__(self, weight_fw_fs, _plot=True):
        self.weight_fw_fs = weight_fw_fs
        self.fw_us = FW_us()
        self.fw_fs = FW_fs()
        self._set_data(_plot)
    def __call__(self, parameters):
        ps = []
        for i, a in enumerate(self.actualizers):
            ps.append(a(parameters[i]))
        er = {"noise_us": self.fw_us(ps) - self.data_us}
        er.update({"noise_fs": self.weight_fw_fs * self.fw_fs(ps) - self.data_fs})
        return er
    def set_actualizers(self, actualizers):
        assert len(actualizers) == 2 # for A and B
        self.actualizers = actualizers
    def _set_data(self, _plot=True):
        np.random.seed(6174)
        self.data_us = self.fw_us([A, B]) + np.random.normal(0, sd_us, len(us))
        np.random.seed(1234)
        self.data_fs = self.weight_fw_fs * ( self.fw_fs([A, B]) + np.random.normal(0, sd_fs, len(fs)) )
        if _plot:
            plt.figure()
            plt.suptitle(f"Noisy data_us - weight_fs={self.weight_fw_fs}")
            plt.subplot(1,2,1)
            plt.plot(self.data_us)
            plt.subplot(1,2,2)
            plt.plot(self.data_fs)
            plt.show()
    
    def jacobian(self, parameters, _fact=-1):
        """
        _fact=-1 is due to implementation of vb.bayes.
        """
        ps = []
        ds = []
        for i, a in enumerate(self.actualizers):
            ps.append(a(parameters[i]))
            ds.append(a.diff(parameters[i]))
        j = {"noise_us": _fact * (ds * self.fw_us.jac(ps))}
        j.update({"noise_fs": _fact * self.weight_fw_fs * (ds * self.fw_fs.jac(ps))})
        return j

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
    test_jacobian_fw()    
    test_jacobian_me()    
    unittest.main()
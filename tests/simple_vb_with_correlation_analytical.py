#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 2021

@author: ajafari
"""

import numpy as np
import matplotlib.pyplot as plt
import unittest
import bayem.correlation as bc

import os, sys
sys.path.append(os.path.dirname(__file__))
from simple_vb_with_correlation import free_energy, vba

param = [5] # mio_target (mean)
param0 = [2] # mio prior
param_prec = 0.001
N = 100
L = 2
def mio(theta):
    return np.full(N, theta[0])

class MeanError:
    def __init__(self, data):
        self.data = data
    def __call__(self, theta):
        k = mio(theta) - self.data
        return k, np.ones((N, 1))

xs = np.linspace(0, L, N)
perfect_data = mio(param)
noise_std = 0.2
correlation_level = 3
C = bc.cor_exp_1d(xs, correlation_level * L / N)
Cinv = bc.inv_cor_exp_1d(xs, correlation_level * L / N)

np.random.seed(6174)
correlated_noise = np.random.multivariate_normal(
    np.zeros(len(xs)),
    C * noise_std ** 2,
)
correlated_data = perfect_data + correlated_noise
f = MeanError(correlated_data)

def do_vb(C_inv=None, s0=1e6, c0=1e-6):
    m0 = np.array(param0)
    L0 = np.array([[param_prec]])
    print('-------------------------- VB started ... ')
    info = vba(f, m0, L0, C_inv=C_inv, s0=s0, c0=c0, update_noise=False)
    for what, value in info.items():
        print(what, value)
    noise_prec_mean = info["shape"] * info["scale"]
    print(f"Noise-Std from mean of identified precision: {noise_prec_mean ** (-0.5)} .")
    return info

def plot_posteriors(infos, labels):
    from scipy.stats import norm
    stds = []
    means = []
    for info in infos:
        stds.append(info["precision"][0, 0] ** (-0.5))
        means.append(info["mean"][0])
    stds_max = max(stds)
    _min = min(param[0], min(means))
    _max = max(param[0], max(means))
    xs = np.linspace(_min - 3 * stds_max, _max + 3 * stds_max, 1000)
    plt.figure()
    for i, info in enumerate(infos):
        dis = norm(means[i], stds[i])
        pdfs = dis.pdf(xs)
        plt.plot(xs, pdfs, label=labels[i])
    plt.axvline(x=param[0], ls="-", color="blue")
    plt.legend()
    plt.title("Posterior of parameter")

    plt.show()

def main(_plot=True):
    if _plot:
        plt.figure()
        plt.plot(perfect_data, label="perfect", marker="*", linestyle="")
        plt.plot(correlated_data, label="noisy", marker="+", linestyle="")
        plt.title("Data")
        plt.legend()
        plt.show()

    # A suitable prior noise that will NOT be updated in VB !
    from bayem.distributions import Gamma
    noise_precision_mean = 1/noise_std**2 # should be equal to target value
    noise_precision_std = noise_precision_mean / (1e6)
        # Interestingly, this does not play any role in the inferred parameters,
        # BUT does change the converged Free energy, so, we set it to a very
        # small value to fulfil as much as possible the assumption of the analytical
        # solution: the noise model (precision) is a known constant !
    gg = Gamma.FromMeanStd(noise_precision_mean, noise_precision_std)
    c0 = gg.shape
    s0 = gg.scale
    
    info = do_vb(s0=s0, c0=c0) # with no correlation
    info2 = do_vb(C_inv=Cinv, s0=s0, c0=c0) # with target correlation
    
    # Analytical by extension of eqs. 9 and 10 of http://gregorygundersen.com/blog/2020/11/18/bayesian-mvn/
    info3 = {}
    Sig_inv = Cinv.todense() * (noise_std ** (-2))
    M = np.sum(Sig_inv) + param_prec
    b = (correlated_data @ np.sum(Sig_inv, axis=1) )[0,0] + param0[0] * param_prec
    mio = b / M
    info3['mean'] = np.array([mio])
    info3['precision'] = np.array([[M]])        
    
    plot_posteriors([info, info2, info3], ['Without correlation', 'With target correlation', 'Analytical'])
    
    err_mean = abs(info2['mean'] - info3['mean']) 
    err_precision = abs(info2['precision'] - info3['precision']) 

    assert err_mean<1e-12
    assert err_precision<5e-12

if __name__ == "__main__":
    main()

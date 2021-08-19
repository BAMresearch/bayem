#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 2021

@author: ajafari

This is to test whether prescribing noise can improve
    the convergence and/or results of the VB inference
    of a simple linear regression model.
"""

import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bayes'))

import copy
import numpy as np
import matplotlib.pyplot as plt
from bayes import vb

## PARAMETERs
E = 1290.0
n_data = 50

## DATA
Eps = 0.02 # perfect strain
eps = n_data * [Eps]
np.random.seed(13)
std_noise = Eps * 0.02
noise = np.random.normal(0., std_noise, n_data)
eps_d = eps + noise # noisy data

F = E * Eps
Fs = n_data * [F]
std_noise_F = F * 0.05
np.random.seed(19)
noise_F = np.random.normal(0., std_noise_F, n_data)
Fs_d = Fs + noise_F
print(f"The standard deviation of the target noise (in residuals) = " + "{:.3e}".format(std_noise_F))

class ME_LinearElastic:
    def __init__(self, eps, Fs):
        self.eps = np.array(eps)
        self.Fs = np.array(Fs)
        self.n_evals = 0
    
    def __call__(self, X):
        EE = X[0]
        residuals = EE * self.eps - self.Fs
        self.n_evals += 1
        return {'res': residuals}
    def jac(self, X, _factor=1):
        EE = X[0]
        return {'res': (_factor * self.eps).reshape((-1,1))}

def priors_femu_f(_update_noise=True):
    # priors
    prior_means = [0.7 * E]
    prior_stds = [0.3 * E]
    precisions = [1.0 / (s**2) for s in prior_stds]
    prior_mvn = vb.MVN(prior_means, np.diag(precisions))
    noise0 = dict()
    update_noise = dict()
    (sh0, sc0) = (0.9361160686609722, 9.362203257716851) # based on expected std of residuals (i.e. std_noise_F)
    noise0['res'] = vb.Gamma(shape=np.array(sh0), scale=np.array(sc0))
    update_noise['res'] = _update_noise
    return prior_mvn, noise0, update_noise

def do_vb(me, prior_mvn, noise0, update_noise, tolerance=1e-2):
    me.n_evals = 0
    
    ## add a minus sign into the jacobian for the VB implementation
    me.jacobian = lambda x: me.jac(x, _factor=-1)
    
    noise_first = False
    # noise_first = True
    # _LM = True
    _LM = False
    vb_outputs = vb.variational_bayes(me, prior_mvn, tolerance=tolerance, noise_first=noise_first, _LM=_LM, noise0=noise0, update_noise=update_noise)    
    print("Took", me.n_evals, "ME simulations.")
    prm_posterior = vb_outputs.param
    noise_posterior = vb_outputs.noise
    # print(vb_outputs)
    print("Prior 'std' of the noises according to the 'mean' of the prior gamma distributions:\n", [(1./n.mean)**0.5 for n in noise0.values()])
    print("Inferred 'std' of the noises according to the 'mean' of the inferred gamma distributions:\n", [(1./n.mean)**0.5 for n in noise_posterior.values()])
    plt.plot(vb_outputs.free_energies, marker='.')
    plt.title('Free energies')
    plt.show(block=False)
    print(vb_outputs)
    return vb_outputs

if __name__ == "__main__":
    ## MODEL ERROR
    me = ME_LinearElastic(eps=eps_d, Fs=Fs_d)
    
    ## PRIORs
    prior_mvn1, noise01, update_noise1 = priors_femu_f(_update_noise=True) # with update of noise of residual
    vb_outputs1 = do_vb(me, prior_mvn1, noise01, update_noise1, tolerance=1e-3)
    prior_mvn2, noise02, update_noise2 = priors_femu_f(_update_noise=False) # with NO update of noise of residual
    vb_outputs2 = do_vb(me, prior_mvn2, noise02, update_noise2, tolerance=1e-3)

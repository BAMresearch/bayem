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
    def __init__(self, eps, Fs, fix_offset=0.0):
        self.eps = np.array(eps)
        self.Fs = np.array(Fs)
        self.fix_offset = fix_offset
        self.n_evals = 0
    
    def __call__(self, X):
        EE = X[0]
        residuals = EE * self.eps + self.fix_offset - self.Fs
        self.n_evals += 1
        return {'res': residuals}
    def jac(self, X, _factor=1):
        return {'res': (_factor * self.eps).reshape((-1,1))}

class ME_LinearElastic_WithOffset:
    def __init__(self, eps, Fs):
        self.eps = np.array(eps)
        self.Fs = np.array(Fs)
        self.ones = np.ones((len(self.eps),))
        self.n_evals = 0
    
    def __call__(self, X):
        EE = X[0]
        b = X[1] # offset
        residuals = (EE * self.eps + b * self.ones) - self.Fs
        self.n_evals += 1
        return {'res': residuals}
    def jac(self, X, _factor=1):
        return {'res': _factor * np.concatenate((self.eps.reshape((-1,1)), self.ones.reshape((-1,1))), axis=1) }

def set_priors(_update_noise=True, _offset=False):
    # priors
    prior_means = [0.7 * E]
    prior_stds = [0.3 * E]
    if _offset:
        prior_means += [0]
        prior_stds += [F/10]
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
    ## Effect of doing or skipping update of Noise
    me = ME_LinearElastic(eps=eps_d, Fs=Fs_d)
    prior_mvn1, noise01, update_noise1 = set_priors(_update_noise=True) # with update of noise of residual
    vb_outputs1 = do_vb(me, prior_mvn1, noise01, update_noise1, tolerance=1e-3)
    prior_mvn2, noise02, update_noise2 = set_priors(_update_noise=False) # with NO update of noise of residual
    vb_outputs2 = do_vb(me, prior_mvn2, noise02, update_noise2, tolerance=1e-3)
    
    ## Effect of doing or skipping identification of Offset (with noise being identified)
    me_3 = ME_LinearElastic(eps=eps_d, Fs=Fs_d, fix_offset=F/20)
    me_4 = ME_LinearElastic_WithOffset(eps=eps_d, Fs=Fs_d)
    prior_mvn3, noise03, update_noise3 = set_priors(_update_noise=True, _offset=False) # without identification of Offset (The fixed offset is Nonzero, though).
    vb_outputs3 = do_vb(me_3, prior_mvn3, noise03, update_noise3, tolerance=1e-3)
    prior_mvn4, noise04, update_noise4 = set_priors(_update_noise=True, _offset=True) # with identification of Offset
    vb_outputs4 = do_vb(me_4, prior_mvn4, noise04, update_noise4, tolerance=1e-3)
    

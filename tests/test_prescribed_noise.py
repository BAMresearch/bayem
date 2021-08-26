#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 2021

@author: ajafari
"""

import unittest
import numpy as np
from bayes import vb
import taralli as trl

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

def print_vb_results(me, vb_outputs, noise0):
    print("Took", me.n_evals, "ME simulations.")
    print(vb_outputs)
    print("Prior 'std' of the noises according to the 'mean' of the prior gamma distributions:\n", [(1./n.mean)**0.5 for n in noise0.values()])
    print("Inferred 'std' of the noises according to the 'mean' of the inferred gamma distributions:\n", [(1./n.mean)**0.5 for n in vb_outputs.noise.values()])

def taralli_solve(me, prior_pars, noise0, _update_noise, _print):
    pass # to be developed

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
        means0 = [0.7 * A, 1.3 * B]
        stds0 = [0.3 * A, 0.3 * B]
        precisions0 = [1.0 / (s**2) for s in stds0]
        self.prior_pars = vb.MVN(means0, np.diag(precisions0))
        ## PRIOR noise
        self.noise0 = dict()
        (sh0, sc0) = ([0.896682869603346], [0.0022557636078939726]) # based on std of Ys
        self.noise0['res'] = vb.Gamma(shape=np.array(sh0), scale=np.array(sc0))
    
    def test_with_noise(self, tolerance=1e-4, _print=True):
        ## VB solution
        update_noise = {'res': True}
        vb_outputs = vb.variational_bayes(self.model, self.prior_pars, tolerance=tolerance, noise0=self.noise0, update_noise=update_noise)    
        if _print:
            print_vb_results(self.model, vb_outputs, self.noise0)
        ## TARALLI (Sampling) solution
        taralli_solve(self.model, self.prior_pars, self.noise0, _update_noise=True, _print=_print)
    
    def test_without_noise(self, tolerance=1e-4, _print=True):
        ## VB solution
        update_noise = {'res': False}
        vb_outputs = vb.variational_bayes(self.model, self.prior_pars, tolerance=tolerance, noise0=self.noise0, update_noise=update_noise)    
        if _print:
            print_vb_results(self.model, vb_outputs, self.noise0)
        ## TARALLI (Sampling) solution
        taralli_solve(self.model, self.prior_pars, self.noise0, _update_noise=False, _print=_print)
    
if __name__ == "__main__":
    unittest.main()
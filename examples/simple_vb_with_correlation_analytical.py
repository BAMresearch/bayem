#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 2021

@author: ajafari
"""

import numpy as np
import matplotlib.pyplot as plt

import bayem.vba as vba
import bayem.distributions as bd
import bayem.correlation as bc

def mio(theta):
    return np.full(N, theta[0])

class MeanError:
    def __init__(self, data):
        self.data = data
    def __call__(self, theta):
        k = mio(theta) - self.data
        return k, np.ones((N, 1))

N = 100
L = 2
param = [5] # mio_target (mean)
param0 = [2] # mio prior
param_prec0 = 0.001
xs = np.linspace(0, L, N)

noise_std = 0.2
correlation_level = 3
cor_length = correlation_level * L / N
noise_cov = bc.cor_exp_1d(xs, cor_length) * noise_std ** 2
cov_inv = bc.inv_cor_exp_1d(xs, cor_length)

perfect_data = mio(param)
np.random.seed(6174)
correlated_noise = np.random.multivariate_normal(
    np.zeros(len(xs)),
    noise_cov ,
)
correlated_data = perfect_data + correlated_noise
plt.figure()
plt.plot(perfect_data, label="perfect", marker="*", linestyle="")
plt.plot(correlated_data, label="noisy", marker="+", linestyle="")
plt.title("Data")
plt.legend()
plt.show()

f = MeanError(correlated_data)

# PRIOR NOISE that will NOT be updated in VB !
noise_precision_mean = 1/noise_std**2 # should be equal to target value
noise_precision_std = noise_precision_mean / (1e4)
    # Interestingly, this does not play any role in the inferred parameters,
    # BUT does change the converged Free energy, so, we set it to a very
    # small value to fulfil as much as possible the assumption of the analytical
    # solution: the noise model (precision) is a known constant !
from bayem.distributions import Gamma
noise0 = Gamma.FromMeanStd(noise_precision_mean, noise_precision_std)

def do_vb(cov_inv=None):
    m0 = np.array(param0)
    L0 = np.array([[param_prec0]])
    prior_mvn = bd.MVN(m0, L0)
    cov_log_det = None if cov_inv is None else (-bc.sp_logdet(cov_inv))
    vb_results = vba(f=f, x0=prior_mvn, noise0=noise0, cov_inv=cov_inv, cov_log_det=cov_log_det \
                     , jac=True, update_noise=False, maxiter=100, tolerance=1e-8, store_full_precision=True)
    return vb_results

def likelihood_times_prior(theta, prec):
    e = f([theta])[0]
    _N = len(e)
    _c = np.sqrt( np.linalg.det(prec) / ((2*np.pi)**_N))
    L = _c * np.exp(-0.5*e.T@prec@e)
    _c2 = np.sqrt( abs(param_prec0) / (2*np.pi))
    P = _c2 * np.exp(-0.5*param_prec0*(theta-param0[0])**2)
    return L * P

def get_analytical_inference():
    ##### POSTERIOR #####
    # Analytical posterior by extension of eqs. 9 and 10 of
    # http://gregorygundersen.com/blog/2020/11/18/bayesian-mvn/
    results_analytic = {}
    Sig_inv = cov_inv.todense() * (noise_std ** (-2))
    M = np.sum(Sig_inv) + param_prec0
    b = (correlated_data @ np.sum(Sig_inv, axis=1) )[0,0] + param0[0] * param_prec0
    mio = b / M
    results_analytic['mean'] = np.array([mio])
    results_analytic['precision'] = np.array([[M]])
    
    ##### LOG-EVIDENCE #####
    ### ANALYTICAL (1) by adaptation of eq (3) in:
    # https://www.econstor.eu/bitstream/10419/85883/1/02084.pdf
    COV = noise_cov
    exponent = (
        -1
        / 2
        * (
            correlated_data.T @ np.linalg.inv(COV) @ correlated_data
            + param0[0] ** 2 * param_prec0
            - mio ** 2 * M
        )
    )
    z = (
        (2 * np.pi) ** (-N / 2)
        * np.linalg.det(COV) ** (-0.5)
        / M**0.5
        * param_prec0 ** 0.5
        * np.exp(exponent)
    )
    logz = np.log(z)
    
    #### ANALYTICAL (2) based on http://gregorygundersen.com/blog/2020/11/18/bayesian-mvn/
    # ---- Not working correctly ... ????!!!!
    # log_ev = np.log( np.sqrt( np.linalg.det(Sig_inv)/((2*np.pi)**N) )  *  np.sqrt( np.abs(param_prec0)/(2*np.pi) ) )
    # _c = (correlated_data @ Sig_inv @ correlated_data)[0,0] + param_prec0 * (param0[0]**2)
    # log_ev += np.log(np.sqrt(2*np.pi)) + (b*b/2/M + _c) - np.log(np.sqrt(M))
    
    #### NUMERICAL (directly from definition)
    from scipy.integrate import quad
    _int_min = -8.0 # =2-10, where 2 is prior mean (at which prior pdf is maximum)
    _int_max = 15.0 # =5+10, where 5 is true mean (at which likelihood is maximum)
        # Setting these integral limits is quite sensitive ...
    log_ev_num, log_err = np.log( quad(likelihood_times_prior, _int_min, _int_max, args=(Sig_inv)\
                            , epsrel=1e-16, epsabs=1e-16, maxp1=1e6) )
    
    return results_analytic, logz, log_ev_num

def plot_posteriors(vb_resultss, labels, results_analytic):
    from scipy.stats import norm
    stds = []
    means = []
    for vb_results in vb_resultss:
        stds.append(vb_results.param.precision[0, 0] ** (-0.5))
        means.append(vb_results.param.mean[0])
    stds_max = max(stds)
    _min = min(param[0], min(means))
    _max = max(param[0], max(means))
    xs = np.linspace(_min - 3 * stds_max, _max + 3 * stds_max, 1000)
    plt.figure()
    for i, vb_results in enumerate(vb_resultss):
        dis = norm(means[i], stds[i])
        pdfs = dis.pdf(xs)
        plt.plot(xs, pdfs, label=labels[i])
    plt.axvline(x=param[0], ls="-", color="blue")
    plt.legend()
    plt.title("Posterior of parameter")

    plt.show()

def study_posterior_and_F():
    
    ##### INFERENCEs #####
    vb_results = do_vb() # with no correlation
    vb_results2 = do_vb(cov_inv=cov_inv) # with target correlation
    results_analytic, logz, log_ev_num = get_analytical_inference()
    
    plot_posteriors([vb_results, vb_results2], ['Without correlation', 'With target correlation'] \
                        , results_analytic)
    
    ##### CHECKs #####
    err_mean = abs(vb_results2.param.mean - results_analytic['mean']) 
    err_precision = abs(vb_results2.param.precision - results_analytic['precision']) 
    err_log_ev = abs((vb_results2.f_max - log_ev_num)/log_ev_num)

    print(f"--------------------------------------------------- \n\
--------------------------------------------------- \n\
------- Free energy (VB with correlation) = {vb_results2.f_max} \n\
------- Log-evidence analytically         = {logz} \n\
------- Log-evidence numerically computed = {log_ev_num} .")
    assert err_mean<1e-12
    assert err_precision<1e-12
    assert err_log_ev<1e-7

def study_correlation_length():
    Fs = []
    _factors = np.linspace(0.5, 1.5, 20+1)
        # to scale the target correlation as the prescribed correlation used for inference
    for _f in _factors:
        cov_inv_0 = bc.inv_cor_exp_1d(xs, cor_length * _f)
        Fs.append(do_vb(cov_inv=cov_inv_0).f_max)
    F_max = max(Fs)
    base_factor_zone = 3.0
    dF_zone = np.log(base_factor_zone)
    plt.figure()
    plt.plot(_factors, Fs, linestyle='', marker='*')
    plt.vlines(1.0, min(Fs), max(Fs))
    plt.fill_between(_factors, F_max - dF_zone, F_max, alpha=0.3)
    plt.text((max(_factors)+1.0)/2.2, F_max - dF_zone/2, s=f"Bayes factor < {base_factor_zone}")
    plt.xlim([_factors[0], _factors[-1]])
    plt.ylim([min(Fs), min(Fs)+(max(Fs)-min(Fs))*1.05])
    plt.xlabel('Prescribed Cor. Length / Target Cor. Length')
    plt.ylabel('Free Energy')
    plt.title('Free energy vs. prescribed correlation length')
    plt.show()
    
if __name__ == "__main__":
    study_posterior_and_F()
    study_correlation_length()

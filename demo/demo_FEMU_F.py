#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 2021

@author: ajafari

This is a demo example to illustrate identification based on the FEMU-F model error
    in the form of [F-f_int , ys-Ys_d]
    with:
        F = external forces (constant)
        f_int = internal forces (at each element)
        Ys_d = Data of U
    The inputs of this model error are:
        - E (Young modulus)
        - ys: imposed ys
"""

import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bayes'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'demo'))


from bayes import vb
from all_helpers import *

_format = '.png'
dpi = 300
sz = 14

## PARAMETERs
E = 129.0
n_data = 50 + 1
L = 13
xs = np.linspace(0, L, n_data)
# xs = np.array([L/(1+np.exp(-a)) for a in np.array(xs)-L/2]) # a non-uniform mesh (inspired by logistic linear regression)
dls = [xs[i+1] - xs[i] for i in range(n_data - 1)]

## DATA
w_pnl = 1.0e0 # a weight factor for penalty term (of defomation deviation)
Ys = 0.02 * xs # perfect data (deformation)
Ys_max = max(Ys)
np.random.seed(13)
std_noise_raw = Ys_max * 0.02
noise = np.random.normal(0., std_noise_raw, n_data)
Ys_d = Ys + noise # noisy data
std_noise = w_pnl * std_noise_raw
F = E * np.max(Ys) / L
Fs = [F for i in range(n_data-1)]

class DemoME_FEMU_F:
    def __init__(self, data, _dict_out=True, w_pnl=w_pnl):
        self.data = data
        self._dict_out = _dict_out
        self.n_evals = 0
        self.w_pnl = w_pnl # a weight factor for penalty term (of defomation deviation)
        self._type = 'FEMU-F'
    
    def __call__(self, E_ys):
        EE = E_ys[0]
        ys = E_ys[1:]
        penalty = self.w_pnl * (ys - self.data)
        residuals = np.array([EE*(ys[i+1]-ys[i])/dls[i] - F for i in range(n_data-1)])
        self.n_evals += 1
        if self._dict_out:
            return {'res': residuals, 'penalty': penalty}
        else:
            return np.concatenate((residuals, penalty))
    def jac(self, E_ys, _factor=1):
        EE = E_ys[0]
        ys = E_ys[1:]
        j11 = np.array([(ys[i+1]-ys[i])/dls[i] for i in range(n_data-1)]) # d_residuals_d_E
        j11 = j11.reshape((n_data-1, 1))
        j12 = np.zeros((n_data-1, n_data)) # d_residuals_d_ys
        for i in range(n_data-1):
            j12[i,i] = -EE / dls[i]
            j12[i,i+1] = EE / dls[i]
        j21 = np.zeros((n_data, 1)) # d_penalty_d_E
        j22 = self.w_pnl * np.eye(n_data)
        J1 = np.append(j11, j12, axis=1)
        J2 = np.append(j21, j22, axis=1)
        if self._dict_out:
            J = {'res': _factor * J1, 'penalty': _factor * J2}
        else:
            J = np.append(J1, J2, axis=0)
        return J

def priors_femu_f(update_of_zero_target_noises=False):
    # priors
    prior_means = [0.5 * E] + n_data * [Ys_max/2]
    prior_stds = [0.3 * E] + n_data * [Ys_max/1.5]
    precisions = [1.0 / (s**2) for s in prior_stds]
    prior_mvn = vb.MVN(prior_means, np.diag(precisions))
    noise0 = dict()
    update_noise = dict()
    percentile0 = 0.01
    ss = std_noise_raw * (E / np.min(dls)) / 100 # expected std of residuals
    sh0, sc0 = gamma_pars_of_precisions_from_stds(stds=[[ss/5, ss*5]], percentile0=percentile0) # for residuals (no noise in data)
    sh1, sc1 = gamma_pars_of_precisions_from_stds(stds=[[std_noise/2, std_noise*2]], percentile0=percentile0) # for penalty (noise in Ys_d)
    # sh1, sc1 = gamma_pars_of_precisions_from_stds(stds=[[std_noise/1.05, std_noise*1.05]], percentile0=percentile0) # for penalty (noise in Ys_d)
    noise0['res'] = vb.Gamma(shape=np.array(sh0), scale=np.array(sc0))
    noise0['penalty'] = vb.Gamma(shape=np.array(sh1), scale=np.array(sc1))
    update_noise['res'] = update_of_zero_target_noises
    update_noise['penalty'] = True
    return prior_mvn, noise0, update_noise

def do_vb(me, prior_mvn, noise0, update_noise, _path, tolerance=1e-2):
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
    plt.savefig(_path + 'Free_energies' + _format, bbox_inches='tight', dpi=dpi)
    plt.show(block=False)
    return vb_outputs

def pp_vb(me, vb_outputs, prior_mvn, noise0, _path):
    base_tit = 'femu-f - VB'
    pr_m = prior_mvn.mean
    pr_std = prior_mvn.std_diag
    post = vb_outputs.param
    post_m = post.mean
    post_std = post.std_diag
    
    from scipy.stats import norm
    pr_E = norm(pr_m[0], pr_std[0])
    post_E = norm(post_m[0], post_std[0])
    plot_pdfs([pr_E, post_E], labels=['Prior', 'Posterior'], colors=[COLORs['prior'], COLORs['posterior']] \
              , target=E, tit=base_tit + " - Young modulus (E)", xl="E", _path=_path, _name=base_tit + ' - E_modulus', _format=_format, dpi=dpi, sz=sz)
    
    pr_m_ys = pr_m[1:]
    pr_std_ys = pr_std[1:]
    post_m_ys = post_m[1:]
    post_std_ys = post_std[1:]
    others = {}
    _plots = []
    _markers = []
    _colors = []
    _perfect = Ys
    _perfect_l = 'perfect'
    _data = Ys_d
    yl = 'Deformation'
    others.update({_perfect_l: _perfect})
    others.update({'data': _data})
    _plots.append(plt.plot)
    _plots.append(plt.scatter)
    _markers.append(MARKERs['perfect'])
    _markers.append(MARKERs['data'])
    _colors.append(COLORs['perfect'])
    _colors.append(COLORs['data'])
    tit = base_tit + ' - Prior deformations'
    plot_box_normal_distribution(pr_m_ys, pr_std_ys, label='prior mean' \
                                  , others=others, colors=_colors, plot_types=_plots, markers=_markers \
                                      , tit=tit, xl='x', yl=yl \
                                          , mean_color=COLORs['prior'], mean_marker=MARKERs['prior'], label_means=False \
                                              , _path=_path, _name=tit, _format=_format, dpi=dpi, sz=sz)
    tit = base_tit + ' - Posterior deformations'
    plot_box_normal_distribution(post_m_ys, post_std_ys, label='posterior mean' \
                                  , others=others, colors=_colors, plot_types=_plots, markers=_markers \
                                      , tit=tit+'\nAverage precision: '+"{:.2e}".format(np.mean(post_std_ys)**(-2)), xl='x', yl=yl \
                                          , mean_color=COLORs['posterior'], mean_marker=MARKERs['posterior'], label_means=False \
                                              , _path=_path, _name=tit, _format=_format, dpi=dpi, sz=sz)
    plt.figure()
    plt.plot(_data, label='data', marker=MARKERs['data'], color=COLORs['data'], linestyle='')
    plt.plot(post_m_ys, label='posterior mean', marker=MARKERs['posterior'], color=COLORs['posterior'], linestyle='')
    _err_ = _data - post_m_ys
    plt.title(base_tit + ' - Deformations\n' + 'Precision of the gaps: ' + "{:.2e}".format(np.std(_err_)**(-2)))
    plt.xlabel('x'); plt.ylabel(yl)
    plt.legend()
    plt.savefig(_path + base_tit + ' - PosteriorMeanVSdata' + _format, bbox_inches='tight', dpi=dpi)
    plt.show(block=False)
    plt.figure()
    plt.plot(_perfect, label=_perfect_l, marker='+', color=COLORs['perfect'], linestyle='')
    plt.plot(post_m_ys, label='posterior mean', marker=MARKERs['posterior'], color=COLORs['posterior'], linestyle='')
    _err_ = _perfect - post_m_ys
    plt.title(base_tit + ' - Deformations\n' + 'Precision of the gaps: ' + "{:.2e}".format(np.std(_err_)**(-2)))
    plt.xlabel('x'); plt.ylabel(yl)
    plt.legend()
    plt.savefig(_path + base_tit + ' - PosteriorMeanVSPerfect' + _format, bbox_inches='tight', dpi=dpi)
    plt.show(block=False)
    
    from scipy.stats import gamma
    interval_prob = 0.9
    target_noises_sds = {'res': 0, 'penalty': std_noise}
    target_noises = {}
    for k, n0 in noise0.items():
        sh_pr = n0.shape
        sc_pr = n0.scale
        n1 = vb_outputs.noise[k]
        sh_post = n1.shape
        sc_post = n1.scale
        g_pr = gamma(a=sh_pr, scale=sc_pr)
        g_post = gamma(a=sh_post, scale=sc_post)
        if np.linalg.norm(target_noises_sds[k])!=0.0:
            target = 1.0 / (target_noises_sds[k]**2) # convert standard deviation to precison
        else:
            target = None
        target_noises.update({k: target})
        k1 = k.replace('_', '\_') # avoid plotting issues
        plot_pdfs([g_pr, g_post], labels=['Prior', 'Posterior'], colors=[COLORs['prior'], COLORs['posterior']], target=target, interval_prob=interval_prob \
                  , tit=base_tit + ' - Precision of noise - ' + k1, xl="$\phi_{"+k1+"}$", _path=_path \
                      , _name=base_tit + ' - Noise precision ' + k, _format=_format, dpi=dpi, sz=sz)
    
    ##### Predictive Posterior
    _path_prd = _path + 'predict/'
    make_path(_path_prd)
    me._dict_out = True
    jac_at_post = me.jacobian(post_m)
    me_at_post = me(post_m)
    ## Posterior forces 
    me._dict_out = True
    Rs = me_at_post['res'] # We evaluate ME based on posterior mean (these will be MEANs)
    fs_id = Rs + Fs # residuals + perfect F
    fs_id_stds = gaussian_error_propagator(stds=post_std, jac=jac_at_post['res'], cov=post.cov) # standard deviations
    others = {}
    _plots = []
    _markers = []
    _colors = []
    others.update({'data': Fs})
    _plots.append(plt.scatter)
    _markers.append(MARKERs['data'])
    _colors.append(COLORs['data'])
    tit = base_tit + ' - Forces'
    plot_box_normal_distribution(fs_id, fs_id_stds, label='predictive' \
                                     , others=others, colors=_colors, plot_types=_plots, markers=_markers \
                                      , tit=tit + '\nAverage precision: ' + "{:.2e}".format(np.mean(fs_id_stds)**(-2)), xl='x', yl='F' \
                                          , mean_color=COLORs['predict'], mean_marker=MARKERs['predict'], label_means=False \
                                              , _path=_path_prd, _name=tit, _format=_format, dpi=dpi, sz=sz)
    plt.figure()
    plt.plot(Fs, label='data', marker='.', linestyle='')
    plt.plot(fs_id, label='predictive (means)', marker=MARKERs['predict'], color=COLORs['predict'], linestyle='')
    plt.title(base_tit + ' - Forces\n' + 'Precision of the gaps: ' + "{:.2e}".format(np.std(Rs)**(-2)))
    plt.xlabel('x'); plt.ylabel('F')
    plt.legend()
    plt.savefig(_path_prd + base_tit + ' - Forces_means' + _format, bbox_inches='tight', dpi=dpi)
    plt.show(block=False)
    
    if me.w_pnl!=0:
        ## Posterior displacements
        us_id = me_at_post['penalty'] + Ys_d
        us_id_stds = gaussian_error_propagator(stds=post_std, jac=jac_at_post['penalty'], cov=post.cov) # standard deviations
        others = {}
        _plots = []
        _markers = []
        _colors = []
        others.update({'data': Ys_d})
        _plots.append(plt.scatter)
        _markers.append(MARKERs['data'])
        _colors.append(COLORs['data'])
        tit = base_tit + ' - Us'
        plot_box_normal_distribution(us_id, us_id_stds, label='predictive' \
                                         , others=others, colors=_colors, plot_types=_plots, markers=_markers \
                                          , tit=tit + '\nAverage precision: ' + "{:.2e}".format(np.mean(us_id_stds)**(-2)), xl='x', yl='U' \
                                              , mean_color=COLORs['predict'], mean_marker=MARKERs['predict'], label_means=False \
                                                  , _path=_path_prd, _name=tit, _format=_format, dpi=dpi, sz=sz)
        plt.figure()
        plt.plot(Ys_d, label='data', marker='.', linestyle='')
        plt.plot(us_id, label='predictive (means)', marker=MARKERs['predict'], color=COLORs['predict'], linestyle='')
        plt.title(base_tit + ' - Us\n' + 'Precision of the gaps: ' + "{:.2e}".format(np.std(us_id-Ys_d)**(-2)))
        plt.xlabel('x'); plt.ylabel('U')
        plt.legend()
        plt.savefig(_path_prd + base_tit + ' - Us_means' + _format, bbox_inches='tight', dpi=dpi)
        plt.show(block=False)

if __name__ == "__main__":
    ##### MODEL ERROR ########
    me = DemoME_FEMU_F(Ys_d, w_pnl=w_pnl)
    
    ##### VARIATIONAL BAYESIAN (VB) ########
    me._dict_out = True
    X = [E] + list(Ys)
    J0 = copy.deepcopy(me.jac(X))
    verify_jacobianDict(me, X, J0, tol=1e-8)
    ## PRIORs
    prior_mvn1, noise01, update_noise1 = priors_femu_f(update_of_zero_target_noises=True) # with update of noise of residual
    prior_mvn2, noise02, update_noise2 = priors_femu_f(update_of_zero_target_noises=False) # with NO update of noise of residual
    
    from pathlib import Path
    _path1 = str(Path(__file__).parent) + '/demo_FEMU_F_VB_1/'
    _path2 = str(Path(__file__).parent) + '/demo_FEMU_F_VB_2/'
    make_path(_path1); make_path(_path2)
    
    ## VB & PostProcess Results
    vb_outputs1 = do_vb(me, prior_mvn1, noise01, update_noise1, _path=_path1, tolerance=3e-2)
    pp_vb(me, vb_outputs1, prior_mvn1, noise01, _path=_path1)
    vb_outputs2 = do_vb(me, prior_mvn2, noise02, update_noise2, _path=_path2, tolerance=3e-2)
    pp_vb(me, vb_outputs2, prior_mvn2, noise02, _path=_path2)

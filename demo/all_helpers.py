#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 2021

@author: ajafari
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import copy

COLORs={}
COLORs.update({'prior':'orange'})
COLORs.update({'posterior':'blue'})
COLORs.update({'data':'red'})
COLORs.update({'perfect':'green'})
COLORs.update({'predict':'purple'})

MARKERs = {}
MARKERs.update({'prior':'<'})
MARKERs.update({'posterior':'>'})
MARKERs.update({'data':'.'})
MARKERs.update({'perfect':''})
MARKERs.update({'predict':'d'})

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def verify_jacobianDict(me, X, J0, tol):
    assert type(J0) == dict
    J0_cd = {}
    for k in J0.keys():
        J0_cd[k] = np.zeros_like(J0[k])
    for i, xi in enumerate(X):
        x_minus = copy.deepcopy(X)
        x_plus = copy.deepcopy(X)
        if abs(xi) <= 1e-9: # too close to zero
            x_minus[i] = -1e-10
            x_plus[i] = 1e-10
            dxi = 2e-10
        else:
            dxi = 2e-6 * abs(xi)
            x_minus[i] = xi - dxi/2
            x_plus[i] = xi + dxi/2
        me_minus = me(x_minus)
        me_plus = me(x_plus)
        for k_out in me_plus.keys():
            J0_cd[k_out][:,i] = ((me_plus[k_out] - me_minus[k_out]) / dxi)[:]
    
    for k_out in me_plus.keys():
        A1 = J0[k_out]
        A2=J0_cd[k_out]
        if np.linalg.norm(A1)==0.0 or np.linalg.norm(A2)==0.0:
            reff = 1.0
        else:
            reff = np.linalg.norm(A1)
        err = np.linalg.norm(A1-A2) / reff
        max_err = err
        if not err < tol:
            print("----- k_out: " + k_out + " -----\nThe relative norm-2 error between analytical and CD-computed assembled Jacobians is: " + str(err))
            raise AssertionError("The given arrays are not identical with relative norm-2 error=" + str(err) + " .")

def gamma_pars_of_precisions_from_stds(stds, percentile0=0.05):
    assert (percentile0<1.0 and percentile0>0.0)
    assert all( (len(b)==2 and b[1]>b[0]) for b in stds)
    shapes = []
    scales = []
    for i, std in enumerate(stds):
        min_st = std[0]
        max_st = std[1]
        min_prec = 1 / (max_st ** 2)
        max_prec = 1 / (min_st ** 2)
        _, sh, sc = gamma_parameters_from_percentiles(min_prec, percentile0, max_prec, 1.0-percentile0)
        shapes.append(sh)
        scales.append(sc)
    return shapes, scales

def gamma_parameters_from_percentiles(x1, p1, x2, p2):
    """
    p1, p2: the cumulative probabilities (for example 0.05 and 0.95)
    x1, x2: the corresponding percentiles (variable's values)
    """
    from scipy.stats import gamma
    from scipy import optimize
    # Standardize so that x1 < x2 and p1 < p2
    if p1 > p2:
        (p1, p2) = (p2, p1)
        (x1, x2) = (x2, x1)
    # function to find roots of for gamma distribution parameters
    def objective(shape):
        return gamma.ppf(p2, shape) / gamma.ppf(p1, shape) - x2/x1
    # The objective function we're wanting to find a root of is decreasing.
    # We need to find an interval over which is goes from positive to negative.
    left = right = 1.0
    while objective(left) < 0.0:
        left /= 2
    while objective(right) > 0.0:
        right *= 2
    shape = optimize.bisect(objective, left, right, \
                            xtol=1e-14*min(x1,x2), \
                                rtol=1e-14)
    scale = x1 / gamma.ppf(p1, shape)
    return (gamma(shape, scale=scale), shape, scale)

def plot_pdfs(dists, labels, colors=None, interval_prob=0.99, target=None, tit='', xl='', yl='pdf', _path='', _name='pdf plot', _format='.png', dpi=300, sz=16, _show_values=True):
    """
    "dists" is a list of scipy.stats distributions
    "interval_prob" is the cumulative distribution (central) based on which the plot's interval is set.
    """
    from matplotlib import rc
    rc('text', usetex=True)
    rc('font', size=sz)
    rc('legend', fontsize=sz)
    rc('text.latex', preamble=r'\usepackage{cmbright}')
    
    _name = _name.replace('$', '') # in case of latex formatted name
    if tit is None:
        tit = 'PDF'
    medians = []; ms = []; stds = []; xxs = []
    x_min = 1e20; x_max = -1e20
    for d in dists:
        medians.append(d.median())
        ms.append(d.mean())
        stds.append(d.std())
        xd0, xd1 = d.interval(interval_prob)
        xxs.append(np.linspace(xd0, xd1, 1000))
        x_min = min(x_min, xd0)
        x_max = max(x_max, xd1)
    from math import isnan
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    max_pdf = 0.0
    max_std = 0.0
    if colors is None:
        colors = len(dists) * ['black']
    for d, dist in enumerate(dists):
        if isnan(stds[d]):
            print('WARNING: No pdf-plot for a distribution with an infinite standard deviation !')
        else:
            m_ = np.format_float_scientific(ms[d], unique=False, precision=2)
            median_ = np.format_float_scientific(medians[d], unique=False, precision=2)
            std_ = np.format_float_scientific(stds[d], unique=False, precision=2)
            pdf_ = dist.pdf(xxs[d])
            if _show_values:
                label_ = labels[d] + r": $\begin{array}{rcl} \mbox{median} & \mbox{=} & " + str(median_) + " \\\\ \mbox{mean} & \mbox{=} & " + str(m_) + " \\\\ \mbox{std} & \mbox{=} & " + str(std_) + " \end{array} $"
            else:
                label_ = labels[d]
            plt.plot(xxs[d], pdf_, label=label_, color=colors[d]);
            t_min = 1e20; t_max = -1e20
            max_pdf = max(np.max(pdf_), max_pdf)
            max_std = max(stds[d], max_std)
    if target is not None:
        if type(target)==list or type(target)==np.ndarray:
            for i, t in enumerate(target):
                target_ = np.format_float_scientific(t, unique=False, precision=2)
                plt.bar(x=t, height=1.0*max_pdf, width=max_std*0.02, color=COLORs['perfect'])
                if _show_values:
                    plt.text(t, 0.9*i*max_pdf/(len(target)-1), f"target{i}=\n{target_}", ha='center', va='bottom', fontsize=sz)
                else:
                    plt.text(t, 0.9*i*max_pdf/(len(target)-1), f"target{i}", ha='center', va='bottom', fontsize=sz)
            t_min, t_max = min(target), max(target)
        else:
            target_ = np.format_float_scientific(target, unique=False, precision=2)
            plt.bar(x=target, height=1.0*max_pdf, width=max_std*0.03, color=COLORs['perfect'])
            if _show_values:
                plt.text(target, 0, f"target={target_}", ha='center', va='bottom', fontsize=sz)
            else:
                plt.text(target, 0, f"target", ha='center', va='bottom', fontsize=sz)
            t_min, t_max = target, target
    plt.xticks(np.linspace(min(x_min, t_min), max(x_max, t_max),3), fontsize=sz);
    plt.yticks(np.linspace(0.0, max_pdf, 3), fontsize=sz);
    plt.title(tit, fontsize=sz)
    plt.xlabel(xl, fontsize=sz)
    plt.ylabel(yl, fontsize=sz)
    plt.legend(fontsize=sz)
    plt.tight_layout()
    plt.savefig(_path + _name + _format, bbox_inches='tight', dpi=dpi)
    plt.show(block=False)

def plot_box_normal_distribution(means, stds, label='' \
                                 , others={}, colors=None, plot_types=None, markers=None \
                                     , tit='', xl='', yl='', sz=16 \
                                         , plot_means=True, mean_color='black', mean_marker='v', label_means=True \
                                             , _path='', _name='normal pdf plot', _format='.png', dpi=300):
    _res = 10
    _widths = -(len(means)/-50) * int(30/_res)
    assert(len(means)==len(stds))
    
    xs = range(1, 1+len(means))
    x_res = max(len(means) // _res, 1)
    means = means[::x_res]
    xs = xs[::x_res]
    
    fig = plt.figure()
    
    if others != {}:
        if colors is None:
            colors = len(others) * ['black']
        if plot_types is None:
            plot_types = len(others) * [plt.plot]
        if markers is None:
            markers = len(others) * ['']
        # from matplotlib.pyplot import cm
        # _colors = iter(cm.rainbow(np.linspace(1.0,0.0,len(others))))
        for i, (k, vals) in enumerate(others.items()):
            # if k=='measured' or k=='measure':
            #     plt.scatter(range(1, 1+len(vals)), vals, label=k, color='green', marker='.')
            # else:
            #     plt.plot(range(1, 1+len(vals)), vals, label=k, color=next(_colors))
            plot_types[i](range(1, 1+len(vals)), vals, label=k, color=colors[i], marker=markers[i])
    
    if plot_means:
        plt.scatter(xs, means, marker=mean_marker, label=label, color=mean_color)
        if label_means:
            for x,y in zip(xs,means):
                label = "{:.3f}".format(y)
                # this method is called for each point
                plt.annotate(label, # this is the text
                             (x,y), # this is the point to label
                             textcoords="offset points", # how to position the text
                             xytext=(0,10), # distance from text to points (x,y)
                             ha='center') # horizontal alignment can be left, right or center
    data = [ [means[i], means[i]-2*stds[i], means[i]+2*stds[i]] for i in range(len(means)) ]
    bx = plt.boxplot(data, positions=xs, widths=_widths)
    for b in bx['boxes']:
        b.set(color=mean_color)
    # plt.title('Box (2*sigma) plot of distributions' + tit, fontsize=sz)
    plt.title(tit, fontsize=sz)
    plt.xlabel(xl, fontsize=sz)
    plt.ylabel(yl, fontsize=sz)
    plt.xticks(ticks=xs, labels=xs, fontsize=sz);
    plt.yticks(fontsize=sz);
    plt.legend(fontsize=sz)
    plt.tight_layout()
    plt.savefig(_path + _name + _format, bbox_inches='tight', dpi=dpi)
    plt.show(block=False)
    
def gaussian_error_propagator(stds, jac, cov=None):
    """
    This method returns a vector as the stds. of the outputs of a vector-valued function
    , which is subjected to inputs with the given stds., where the function's Jacobian
    at the corresponding mean values is "jac" (the mean values themselves do not matter).
    """
    if cov is None: # no correlation among the inputs are considered.
        assert len(stds)==jac.shape[1]
        return ( (jac**2) @ (np.array(stds)**2) ) ** 0.5
    else:
        assert cov.shape[0]==cov.shape[1]==jac.shape[1]
        assert np.linalg.norm(cov-cov.T)<1e-9 # covariance matrix must be symmetric for the following formula to be valid.
        return ( np.diag((jac @ cov) @ jac.T) ) ** 0.5
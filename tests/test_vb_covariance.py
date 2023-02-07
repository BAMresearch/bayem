import numpy as np
import unittest
import bayem.vba as vba
import bayem.distributions as bd
import bayem.correlation as bc

def _do_vb(f, prior_mvn, noise0, cov_inv=None \
           , jac=True, update_noise=True, maxiter=100, tolerance=1e-8, _print=False):
    cov_log_det = None if cov_inv is None else (-bc.sp_logdet(cov_inv))
    vb_results = vba(f=f, x0=prior_mvn, noise0=noise0, cov_inv=cov_inv, cov_log_det=cov_log_det \
                     , jac=jac, update_noise=update_noise, maxiter=maxiter, tolerance=tolerance \
                     , store_full_precision=True)
    if _print:
        noise_prec_mean = vb_results.noise.shape * vb_results.noise.scale
        aa = 'without' if cov_inv is None else 'with'
        _msg = f"\n----- VB {aa} covariance of noise:"
        _msg += f"\n\t- Max. free energy = {vb_results.f_max:.2f} ,"
        _msg += f"\n\t- Inferred pars (mean) = {vb_results.param.mean} ,"
        _msg += f"\n\t- Inferred pars (precision) = {vb_results.param.precision} ,"
        _msg += f"\n\t- Inferred noise precision (mean): {noise_prec_mean:.2f} ."
        print(_msg)
    return vb_results

class TestVBCovariance(unittest.TestCase):
    
    def test_vb_cov(self):
        N = 100
        L = 2
        xs = np.linspace(0, L, N)
        def g(theta):
            return theta[1] ** 2 + xs * theta[0]
        class F:
            def __init__(self, data):
                self.data = data
        
            def __call__(self, theta):
                k = g(theta) - self.data
                d_dm = xs
                d_dc = 2 * theta[1] * np.ones_like(xs)
                return k, np.vstack([d_dm, d_dc]).T
        
        param = [5, 7]
        param_prec = 0.001
        noise_std = 0.2
        correlation_level = 5
        cor_length = correlation_level * L / N
        noise_cov = bc.cor_exp_1d(xs, cor_length) * noise_std ** 2
        cov_inv = bc.inv_cor_exp_1d(xs, cor_length)
        
        perfect_data = g(param)
        np.random.seed(6174)
        correlated_noise = np.random.multivariate_normal(
            np.zeros(len(xs)),
            noise_cov ,
        )
        correlated_data = perfect_data + correlated_noise
        f = F(correlated_data)
        
        m0 = np.array([2, 19])
        L0 = np.array([[param_prec, 0], [0, param_prec]])
        prior_mvn = bd.MVN(m0, L0)
        noise0 = bd.Gamma(shape=1e-6, scale=1e6)
        
        vb_results = _do_vb(f=f, prior_mvn=prior_mvn, noise0=noise0)
        vb_results2 = _do_vb(f=f, prior_mvn=prior_mvn, noise0=noise0, cov_inv=cov_inv)
        _factor = 0.5
        vb_results3 = _do_vb(f=f, prior_mvn=prior_mvn, noise0=noise0, cov_inv=cov_inv/_factor)
        
        # In the first scenario we obtain more certain (higher precision) inference of parameters
        # , since we did not account for correlation of data.
        assert vb_results.param.precision[0, 0] > vb_results2.param.precision[0, 0]
        assert vb_results.param.precision[1, 1] > vb_results2.param.precision[1, 1]
    
        ## Since covariance matrix in vb_results3 is half of vb_results2 (only a scaling):
        # 1) The inferred parameters in vb_results2 and vb_results3 must be the same.
        # 2) The inferred noise precision in vb_results3 must be half of vb_results2.
        # This ratio, however, only goes to the mean of the identified noise, meaning that:
        # The scales are different by the factor 2.
        # But the shapes are the same (we only have shift of one distribution from the other).
        # 3) The converged free energy in vb_results2 and vb_results3 is the same.
        assert (
            np.linalg.norm(vb_results3.param.mean - vb_results2.param.mean) / np.linalg.norm(vb_results2.param.mean)
            < 1e-6
        )
        assert (
            np.linalg.norm(vb_results3.param.precision - vb_results2.param.precision)
            / np.linalg.norm(vb_results2.param.precision)
            < 1e-6
        )
        assert abs((vb_results3.noise.shape - vb_results2.noise.shape) / vb_results2.noise.shape) < 1e-6
        assert abs((vb_results3.noise.scale - _factor * vb_results2.noise.scale) / vb_results2.noise.scale) < 1e-6
        assert abs((vb_results3.f_max - vb_results2.f_max) / vb_results2.f_max) < 1e-5
    
    def test_vb_cov_analytic(self):
        N = 100
        L = 2
        xs = np.linspace(0, L, N)
        def mio(theta):
            return np.full(N, theta[0])
        class MeanError:
            def __init__(self, data):
                self.data = data
            def __call__(self, theta):
                k = mio(theta) - self.data
                return k, np.ones((N, 1))
        
        param = [5] # mio_target
        param0 = [2] # prior (mean)
        param_prec0 = 0.001 # prior (precision)
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
        f = MeanError(correlated_data)
        
        m0 = np.array(param0)
        L0 = np.array([[param_prec0]])
        prior_mvn = bd.MVN(m0, L0)
        # PRIOR NOISE that will NOT be updated in VB !
        noise_precision_mean = 1. / noise_std ** 2 # should be equal to target value (and not updated)
        noise_precision_std = noise_precision_mean / (1e4)
            # Interestingly, this does not play any role in the inferred parameters,
            # BUT does change the converged Free energy, so, we set it to a very
            # small value to fulfil as much as possible the assumption of the analytical
            # solution: the noise model (precision) is a known constant !
        noise0 = bd.Gamma.FromMeanStd(noise_precision_mean, noise_precision_std)
        
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
        
        ##### INFERENCEs #####
        vb_results = _do_vb(f=f, prior_mvn=prior_mvn, noise0=noise0) # with no covariance
        vb_results2 = _do_vb(f=f, prior_mvn=prior_mvn, noise0=noise0, cov_inv=cov_inv) # with target covariance
        results_analytic, logz, log_ev_num = get_analytical_inference()
        
        ##### CHECKs #####
        err_mean = abs(vb_results2.param.mean - results_analytic['mean']) 
        err_precision = abs(vb_results2.param.precision - results_analytic['precision']) 
        err_log_ev = abs((vb_results2.f_max - log_ev_num)/log_ev_num)
    
        print(f"\n\
        ----------------------------------------------------------- \n\
        ------- Free energy (VB with covariance) = {vb_results2.f_max} \n\
        ------- Log-evidence analytically         = {logz} \n\
        ------- Log-evidence numerically computed = {log_ev_num} .")
        assert err_mean<1e-12
        assert err_precision<1e-12
        assert err_log_ev<1e-7
    
    
if __name__ == "__main__":
    unittest.main()
    
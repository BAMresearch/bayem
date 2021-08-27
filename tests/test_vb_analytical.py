import numpy as np
import unittest
import bayes.vb

"""
Examples from
http://www.stat.cmu.edu/~brian/463-663/week09/Chapter%2003.pdf
section 3.4 about midterm results of 30 students.
"""

n = 30
np.random.seed(0)
data = np.random.normal(75, 10, size=n)
mx, sigma = np.mean(data), np.std(data)

prior_mean = 70
prior_sd = 5

def model_error(p):
    model = np.ones(n) * p[0]
    return {"error": model - data}

def compute_mean_and_var(vals, pdfs):
    assert len(vals)==len(pdfs)
    # we first integrate the first and last points (based on delta/2)
    m = vals[0] * pdfs[0] * (vals[1]-vals[0]) / 2 \
        + vals[-1] * pdfs[-1] * (vals[-1]-vals[-2]) / 2
    m2 = (vals[0] **2) * pdfs[0] * (vals[1]-vals[0]) / 2 \
        + (vals[-1] **2) * pdfs[-1] * (vals[-1]-vals[-2]) / 2 # second momentum
    for i in range(1, len(vals)-1):
        m += vals[i] * pdfs[i] * (vals[i+1]-vals[i-1]) / 2
        m2 += (vals[i]**2) * pdfs[i] * (vals[i+1]-vals[i-1]) / 2
    var = m2 - m**2
    return m, var

## These values are used for numerical integrations in the following class
_min_m = 70
_max_m = 88
_min_sig = 5
_max_sig = 20

class AnalyticInferMeanAndSigmaFromNoninformativePriors:
    def __init__(self, data=data):
        """
        Example from 
        http://www.stat.cmu.edu/~brian/463-663/week09/Chapter%2003.pdf
        Based on eq. (3.4) of the equation.
        It has nothing about any priors, since it is based on non-informative priors.
        """
        self.data = data
        self.n = len(data)
        self._treat_joint_posterior(self)
    
    def _treat_joint_posterior(self, _plot=True):
        self._const_of_joint_post = None
        self._const_of_joint_post = 1.0 / self._compute_denom_of_joint_post()
        assert abs(1.0- self._compute_denom_of_joint_post()) < 1e-5 # For the second time, the constant must be already 1.
        if _plot:
            from mpl_toolkits.mplot3d import axes3d
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            X, Y = np.mgrid[_min_m:_max_m:200j, _min_sig:_max_sig:200j]
            Z = self._joint_posterior(X, Y)
            ax.plot_surface(X, Y, Z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1)
            plt.title('Analytical Posterior Joint Distribution (pdf)')
            plt.xlabel('mio')
            plt.ylabel('sigma')
            plt.show()
        sigmas, pdf_sigma = self._marginalize_out_mio()
        self.analytical_sig_mean, self.analytical_sig_var = compute_mean_and_var(vals=sigmas, pdfs=pdf_sigma)
        ms, pdf_m = self._marginalize_out_sigma()
        self.analytical_m_mean, self.analytical_m_var = compute_mean_and_var(vals=ms, pdfs=pdf_m)
    
    def _joint_posterior(self, mx, sigma):
        """
        This is formula 3.4 in the document mentioned above.
        Note that, due to noninformative priors behind this formula, we see no prior parameters in it.
        """
        sigma2 = sigma * sigma
        j_post = sigma2**(-1)
        _exp = 0
        for i in range(self.n):
            j_post /= np.sqrt(2*np.pi*sigma2)
            _exp += -((self.data[i] - mx)**2) / (2*sigma2)
        j_post *= np.exp(_exp)
        if self._const_of_joint_post is not None:
            j_post *= self._const_of_joint_post
        return j_post
    
    def _compute_denom_of_joint_post(self):
        from scipy.integrate import dblquad as myinteg2
        ans = myinteg2(func=self._joint_posterior \
                       , a=_min_sig, b=_max_sig, gfun=lambda x: _min_m, hfun=lambda x: _max_m)
            # a, b are the limits for sigma (second argument of func)
            # gfun, hfun are the limits for m (first argument of func)
        return ans[0] # ans[1] is the accuracy
    
    def _marginalize_out_mio(self, _plot=True):
        pdf_sigma = []
        range_sigma = [_min_sig, _max_sig]
        res_sigma = 1000
        sigmas = np.linspace(range_sigma[0], range_sigma[1], res_sigma)
        from scipy.integrate import quad as myinteg
        for s in sigmas:
            pdf_sigma.append(
                myinteg(self._joint_posterior, a=_min_m, b=_max_m, args=(s))[0]
                )
        if _plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(sigmas, pdf_sigma, label='Analytic')
            plt.title('Marginalized pdf of noise model (standard deviation)')
            plt.xlabel('sigma')
            plt.legend()
            plt.show()
        return sigmas, pdf_sigma
    
    def _marginalize_out_sigma(self, _plot=True):
        pdf_m = []
        range_m = [_min_m, _max_m]
        res_m = 1000
        ms = np.linspace(range_m[0], range_m[1], res_m)
        from scipy.integrate import quad as myinteg
        def reverse_inputs(sig, m):
            return self._joint_posterior(m, sig)
        for m in ms:
            pdf_m.append(
                myinteg(reverse_inputs, a=_min_sig, b=_max_sig, args=(m))[0]
                )
        if _plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(ms, pdf_m, label='Analytic')
            plt.title('Marginalized pdf of parameter mean')
            plt.xlabel('m')
            plt.legend()
            plt.show()
        return ms, pdf_m

class Test_VBAnalytic(unittest.TestCase):
    def setUp(self):
        pass

    def test_given_noise(self):
        """
        We infer the distribution mu for a given, fixed noise. This means no 
        update of the gamma distribution within variational_bayes by passing
        the kwarg `update_noise={"error" : False}`

        For the prior N(prior_mean, scale=prior_sd), the parameters of the
        posterior distribution N(mean, scale) read
        """
        denom = n * prior_sd ** 2 + sigma ** 2
        mean = (sigma ** 2 * prior_mean + prior_sd ** 2 * n * mx) / denom
        variance = (sigma ** 2 * prior_sd ** 2) / denom
        scale = variance **0.5

        prior = bayes.vb.MVN(prior_mean, 1.0 / prior_sd ** 2)
        gamma = {"error": bayes.vb.Gamma.FromSD(sigma)}

        result = bayes.vb.variational_bayes(
            model_error, prior, gamma, update_noise={"error": False}
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.param.mean[0], mean)
        self.assertAlmostEqual(result.param.std_diag[0], scale)

    def test_given_mu(self):
        """
        We infer the gamma distribution of the noise precision for a 
        given, fixed parameter mu. This is done by setting a prior with
        a very high precision.

        For a super noninformative noise prior (shape=0, scale->inf), the 
        analytic solution for the INVERSE gamma distribution for the noise
        VARIANCE reads
        """
        a = n / 2
        b = np.sum((data - prior_mean) ** 2) / 2

        """
        The parameters for the corresponding gamma distribution for the 
        PRECISION then read (a, 1/b)
        """
        
        big_but_not_nan = 1e50
        prior = bayes.vb.MVN(prior_mean, big_but_not_nan)
        gamma = {"error": bayes.vb.Gamma(shape=0, scale=big_but_not_nan)}

        result = bayes.vb.variational_bayes(
            model_error, prior, gamma, update_noise={"error": True}
        )
        self.assertTrue(result.success)

        gamma = result.noise["error"]
        self.assertAlmostEqual(gamma.shape, a)
        self.assertAlmostEqual(gamma.scale, 1 / b)

    def test_infer_both(self):
        ## ANALITICAL
        pp = AnalyticInferMeanAndSigmaFromNoninformativePriors(data)
        ## VB
        small_but_not_zero = 1e-50
        big_but_not_nan = 1e50
        prior = bayes.vb.MVN(prior_mean, small_but_not_zero) # noninformative prior parameter
        gamma = {"error": bayes.vb.Gamma(shape=0.0, scale=big_but_not_nan)}  # noninformative prior noise precision

        result = bayes.vb.variational_bayes(
            model_error, prior, gamma, update_noise={"error": True}, tolerance=1e-6
        )
        self.assertTrue(result.success)
        ## COMPARISONs
        # Compare parameter (mean and std)
        self.assertAlmostEqual(pp.analytical_m_mean / result.param.mean[0], 1.0, delta=1e-5)
        self.assertAlmostEqual(pp.analytical_m_var**(0.5) / result.param.std_diag[0], 1.0, delta=2e-2)
        # Compare noise (mean of std)
        self.assertAlmostEqual(pp.analytical_sig_mean / result.noise['error'].mean**(-0.5), 1.0, delta=1e-2)


if __name__ == "__main__":
    import logging

    # logging.basicConfig(level=logging.DEBUG)
    unittest.main()

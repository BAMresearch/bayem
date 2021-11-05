import numpy as np
import scipy.stats
import dynesty
import bayes.vb
import json

n = 30
np.random.seed(0)
data = np.random.normal(75, 10, size=n)
mx, sigma = np.mean(data), np.std(data)

prior_mean = 70
prior_sd = 5
prior = scipy.stats.norm(prior_mean, prior_sd)

# analytic posterior:
denom = n * prior_sd ** 2 + sigma ** 2
mean = (sigma ** 2 * prior_mean + prior_sd ** 2 * n * mx) / denom
variance = (sigma ** 2 * prior_sd ** 2) / denom
scale = variance ** 0.5

def model_error(theta):
    return data - theta

def loglike(theta):
    logpdfs = scipy.stats.norm.logpdf(model_error(theta), scale=sigma)
    return np.sum(logpdfs)

shape = 1e10
vb_result = bayes.vb.variational_bayes(model_error, param0=bayes.vb.MVN(prior_mean, 1/prior_sd**2), noise0=bayes.vb.Gamma(shape=shape, scale=1/shape * 1/sigma**2), update_noise=False)
        
assert abs(vb_result.param.mean[0] - mean) < 1.e-8
assert abs(vb_result.param.std_diag[0] - scale) < 1.e-8
print("analytic VB free energy:", vb_result.f_max)

# sampler = dynesty.DynamicNestedSampler(loglikelihood=loglike, prior_transform=prior.ppf, ndim=1)
# sampler.run_nested()
# print("dnyesty log evidence:   ", sampler.results.logz[-1])
print("dnyesty log evidence:   -116.54005116")

# analytic solution based on 
# https://www.econstor.eu/bitstream/10419/85883/1/02084.pdf eq 3 (whereever that is coming from)
m = (2*np.pi)**(-n/2) * sigma**(-n) * scale/prior_sd * np.exp(-1/2*(np.dot(data, data)/sigma**2 + prior_mean**2/prior_sd**2 - mean**2/scale**2))
logz = np.log(m) # this approach is almost exact, but error prone due to numerical imprecisions
# print(logz)
logz = -n/2 * np.log(2*np.pi) - n * np.log(sigma) + np.log(scale/prior_sd) -1/2*(np.dot(data, data)/sigma**2 + prior_mean**2/prior_sd**2 - mean**2/scale**2)
print("analytic solution:     ", logz) # well, no difference though


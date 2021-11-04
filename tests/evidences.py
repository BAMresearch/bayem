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

vb_result = bayes.vb.variational_bayes(model_error, param0=bayes.vb.MVN(prior_mean, 1/prior_sd**2), noise0=bayes.vb.Gamma(scale=1, shape=1/sigma**2), update_noise=False)
        
assert abs(vb_result.param.mean[0] - mean) < 1.e-8
assert abs(vb_result.param.std_diag[0] - scale) < 1.e-8
print(vb_result.f_max)

sampler = dynesty.DynamicNestedSampler(loglikelihood=loglike, prior_transform=prior.ppf, ndim=1)
# sampler.run_nested(nlive_init=10)
sampler.run_nested()
# with open("result.json", "w") as f:
    # json.dump(sampler.results, f)


# dynesty.plotting.runplot(sampler.results)


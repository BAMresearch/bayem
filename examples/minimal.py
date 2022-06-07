import bayem
import numpy as np

x = np.linspace(0, 1, 1000)
data = 5 * x + 42 + np.random.normal(size=1000)


def k(theta):
    return theta[0] * x + theta[1] - data


wide_prior = bayem.MVN.FromMeanStd([0,0], [1000, 1000])
result = bayem.vba(k, x0=wide_prior)
result.summary()

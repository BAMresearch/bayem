import bayem
import numpy as np

N, sigma = 100, 0.2
x = np.linspace(0, 1, N)
data = 5 * x + 42 + np.random.normal(0, sigma, N)


def k(theta):
    return theta[0] * x + theta[1] - data


prior = bayem.MVN.FromMeanStd([10, 10], [100, 100])
result = bayem.vba(k, x0=prior)
result.summary()

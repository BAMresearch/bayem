[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17805357.svg)](https://doi.org/10.5281/zenodo.17805357)

# Bayesian Inference

~~~py
import bayem
import numpy as np

x = np.linspace(0, 1, 1000)
data = 5 * x + 42 + np.random.normal(size=1000)

def k(theta):
    return theta[0] * x + theta[1] - data

wide_prior = bayem.MVN.FromMeanStd([0,0], [1000, 1000])
result = bayem.vba(k, x0=wide_prior)
result.summary()

# name            median    mean      sd      5%     25%     75%     95%
# ------------  --------  ------  ------  ------  ------  ------  ------
# $\theta_{0}$     4.951   4.951   0.113   4.765   4.875   5.027   5.137
# $\theta_{1}$    41.989  41.989   0.065  41.882  41.945  42.033  42.096
# noise            0.939   0.940   0.042   0.872   0.911   0.968   1.010
~~~

`numpy`-based implementation of an analytical variational Bayes algorithm of

> "Variational Bayesian inference for a nonlinear forward model." 
> 
> Chappell, Michael A., Adrian R. Groves, Brandon Whitcher, and Mark W. Woolrich. 
> IEEE Transactions on Signal Processing 57, no. 1 (2008): 223-236.

with an updated free energy equation to correctly capture the log evidence. The algorithm requires a user-defined model error allowing an arbitrary combination of custom forward models and measured data.

A recently published implementation by the group of Chappell based on `tensorflow` can be found [here](https://github.com/physimals/vaby_avb).

# Installation

~~~sh
pip3 install .
~~~

# Derivation

The derivation of the currently implemented variational method can be found [here](https://bayesianinference.readthedocs.io/en/latest/). To build the website locally, run
~~~sh
$ pip3 install .[doc]
$ doit
~~~

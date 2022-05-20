![Python application](https://github.com/BAMresearch/BayesianInference/workflows/Python%20application/badge.svg?branch=master)

# Bayesian Inference

~~~py
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

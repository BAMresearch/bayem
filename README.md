![Python application](https://github.com/BAMresearch/BayesianInference/workflows/Python%20application/badge.svg?branch=master)

# Bayesian Inference

`numpy`-based implementation of an analytical variational Bayes algorithm of

> "Variational Bayesian inference for a nonlinear forward model." 
> 
> Chappell, Michael A., Adrian R. Groves, Brandon Whitcher, and Mark W. Woolrich. 
> IEEE Transactions on Signal Processing 57, no. 1 (2008): 223-236.

with an updated free energy equation to correctly capture the log evidence. 

The recently published [authors implementation](https://github.com/physimals/vaby_avb) provides a set of models **g(θ)** that can be combined to a time-series-like dataset **y**, forming the model error **k(θ) = g(θ) - y**. The algorithm presented here is based directly on a user-defined model error **k(θ)** to allow a arbitrary combinations of custom forward models and the measured data.

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

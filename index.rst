============================================================
   Derivation of variational Bayesian for nonlinear models
============================================================

Scope
===========================
The derivation procedure for the formulation of the Variational Bayesian (VB) method proposed in the references listed below is addressed here in details together with an improvement in the noise model as compared to the one used in these works; namely, the noise model can adopt a prescribed non-uniform covariance matrix.

*   Chappell, M. A.; Groves, A. R.; Whitcher, B. \& Woolrich, M. W.
    Variational Bayesian Inference for a Nonlinear Forward Model,
    IEEE Transactions on Signal Processing, 2009, 57, 223-236
    https://ieeexplore.ieee.org/document/4625948
*   Chappell, M. A.; Groves, A. R. \& Woolrich, M. W.
    The FMRIB Variational Bayes Tutorial: Variational Bayesian inference for non-linear forward model,
    https://users.fmrib.ox.ac.uk/~chappell/papers/TR07MC1.pdf

Main criterion (maximization of free energy)
============================================
With :math:`\boldsymbol{y}`: measured data, :math:`\boldsymbol{w}=[\boldsymbol{w}_i]`: latent parameters, :math:`P(\boldsymbol{w})`: priors and :math:`q(\boldsymbol{w})`: approximated posteriors, the following free energy must be maximized.

.. math::
	\require{cancel}
    F = \int q(\boldsymbol{w}) \log \left[ \frac{P(\boldsymbol{y}|\boldsymbol{w})\,P(\boldsymbol{w})}{q(\boldsymbol{w})} \right] d\boldsymbol{w} \, .
    :label: eq_free_energy

We use the mean field approximation for posteriors :math:`q(\boldsymbol{w})`:

.. math::
    q(\boldsymbol{w}) = \prod_{i=1}^m q_{i}(\boldsymbol{w}_i) \, ,
    :label: eq_mean_field

and apply the calculus of variations to fulfil the maximization of the free energy. For this purpose, we regroup the free energy defined above for every and each index :math:`i`.

.. math::
    F &= \int q_{i} \, q_{\cancel{i}} \,
    \log\left[P(\boldsymbol{y}|\boldsymbol{w})\, P(\boldsymbol{w})\right]
    - q_{i} \,q_{\cancel{i}} \, \log[q_{i}]
    - q_{i} \, q_{\cancel{i}} \, \log[q_{\cancel{i}}]
    \;d \boldsymbol{w} ; \; \mbox{for each } i \, .\\
    :label: eq_free_energy_expanded

where :math:`_{\cancel{i}}` indicates all indices except :math:`i`, and :math:`q_\cancel{i}` represents the product of all distributions except :math:`q_i`.

We consider the above equation for every and each :math:`_i` and shift integrants corresponding to :math:`\boldsymbol{w}_\cancel{i}` to a separate term (:math:`g`), resulting in the following expression.

.. math::
    F &= \int g\left(\boldsymbol{w}_i, q_{i}(\boldsymbol{w}_i)\right) \;d\boldsymbol{w}_i \, ,\\
    :label: eq_free_energy_compact

with

.. math::
    g\left(\boldsymbol{w}_i, q_{i}(\boldsymbol{w}_i)\right) &=
    \int f\left(\boldsymbol{w}, q(\boldsymbol{w})\right) \;d\boldsymbol{w}_\cancel{i} \, ,\\

where :math:`f\left(\boldsymbol{w}, q(\boldsymbol{w})\right)` is the whole integrant in :eq:`eq_free_energy_expanded`.

From variational calculus, the maximum of F represented in :eq:`eq_free_energy_compact` is the solution of the `Euler-Lagrange equation <https://en.wikipedia.org/wiki/Calculus_of_variations#Euler%E2%80%93Lagrange_equation>`_.

.. math::
    \frac{\partial}{\partial q_i(\boldsymbol{w}_i)} \left[
    g\left(\boldsymbol{w}_i, q_{i}(\boldsymbol{w}_i), q'_{i}(\boldsymbol{w}_i)\right)
    \right]-
    \frac{d}{d\boldsymbol{w}_i}\left\{
    \frac{\partial}{\partial q'_i(\boldsymbol{w}_i)}\left[g(
    \boldsymbol{w}_i, q_{i}(\boldsymbol{w}_i), q'_{i}(\boldsymbol{w}_i))
    \right]
    \right\}&=0.\\

The second term vanishes, since :math:`g` does not depend on :math:`q'_i(\boldsymbol{w}_i)`. Substituting the first
term using the mean field approximation of F yields

.. math::
    0&= \frac{\partial }{\partial q_i} \left[ \int
    q_{i} \, q_{\cancel{i}} \,
    \log\left[P(\boldsymbol{y}|\boldsymbol{w})\, P(\boldsymbol{w})\right]
    - q_{i} \,q_{\cancel{i}} \, \log[q_{i}]
    - q_{i} \, q_{\cancel{i}} \, \log[q_{\cancel{i}}]
    \;d \boldsymbol{w}_{\!\cancel{i}}
    \right]\\
    &= \int  q_{\cancel{i}}\log[P(\boldsymbol{y}|\boldsymbol{w})\,P(\boldsymbol{w})] -
    (q_{\cancel{i}} \,
    \log[q_{i}] + \frac{q_{i} \, q_{\cancel{i}}}{q_{i}}) - q_{\cancel{i}} \,\log[q_{\cancel{i}}] \;
    d\boldsymbol{w}_{\!\cancel{i}}.\\

With the property of a density function to integrate to one, it follows:

.. math::
    0&= \int q_{\cancel{i}}\log[P(\boldsymbol{y}|\boldsymbol{w}) P(\boldsymbol{w})]d\boldsymbol{w}_{\cancel{i}}
    -  \log[q_{i}] - 1 - \int q_{\cancel{i}}\log[q_{\cancel{i}}]d\boldsymbol{w}_{\!\cancel{i}}.\\

Moving the second term to the other side of the equation and realizing that the latter two terms do not depend on :math:`q_{i}`, we obtain the following set of governing equations for the maximization of the free energy.

.. math::
	\log[q_{i}] = \int q_{\cancel{i}}\log \left[ P(\boldsymbol{y}|\boldsymbol{w}) P(\boldsymbol{w}) \right] d\boldsymbol{w}_{\!\cancel{i}} + \mbox{const.} \quad ; \; \mbox{for each } i \, .\\
    :label: eq_qw_main

Note that, :eq:`eq_qw_main` agrees with eq.(5) of Chappell paper:

.. math::
    \log[q_{i}] & \propto \int q_{\cancel{i}}\log[P(\boldsymbol{y}|\boldsymbol{w}) P(\boldsymbol{w})]
    d\boldsymbol{w}_{\!\cancel{i}} \, .\\


Model of noise (error)
======================
We *define* the noise (error) :math:`\boldsymbol{e}` as the gap between the data :math:`\boldsymbol{y}` and the model response :math:`g(\theta)`:

.. math::
	\boldsymbol{e}=\boldsymbol{e}(\theta)=:\boldsymbol{y}-g(\theta) \, ,\\
	:label: eq_e_define

and *model* it through a multivariate normal distribution, i.e.:

.. math::
	\boldsymbol{e} \sim \mathcal{MVN}(\boldsymbol{0}, \Phi^{-1} \textcolor{red}{C_e} ) \quad \rightarrow \quad P(\boldsymbol{e}|\Phi)=\frac{\Phi^{N/2}}{\sqrt{(2\pi)^{N}\textcolor{red}{|C_e|}}}e^{-0.5\,\Phi\,\boldsymbol{e}^T. \textcolor{red}{C_e^{-1}} .\boldsymbol{e}} \; ,
	:label: eq_e_model

where :math:`N` is the length of model error signal and :math:`\textcolor{red}{C_e}` is a known covariance matrix scaled by means of the latent parameter :math:`\Phi` to represent the covariance of the distribution.

With such a setup, the latent parameters will become:

.. math::
	\boldsymbol{w} = [\boldsymbol{w}_{\theta}, \boldsymbol{w}_{\Phi}] \; ,
	:label: eq_w_list

which also implies :math:`i \in \left\{ \theta, \Phi \right\}` in :eq:`eq_mean_field` and :eq:`eq_qw_main`, resulting in the following governing equations for the inference.

.. math::
	\log[q_{\theta}] = \int q_{\Phi} \log \left[ P(\boldsymbol{y}|\boldsymbol{w}) P(\boldsymbol{w}) \right] d\Phi + \mbox{const.} \, .
	:label: eq_qtheta_main

.. math::
	\log[q_{\Phi}] = \int q_{\theta} \log \left[ P(\boldsymbol{y}|\boldsymbol{w}) P(\boldsymbol{w}) \right] d\theta + \mbox{const.} \, .
	:label: eq_qPhi_main

Log likelihood
==============
The likelihood is by definition:

.. math::
	\mbox{Likelihood} =: P(\boldsymbol{y}|\boldsymbol{w}) = P(\boldsymbol{y}|\theta, \Phi) = \underbrace{P(\boldsymbol{y}|\boldsymbol{e}, \theta, \Phi)}_{=1} \, P(\boldsymbol{e}(\theta)|\Phi)=P(\boldsymbol{e}|\Phi) \; .
	:label: eq_likl

According to :eq:`eq_e_model`, the log likelihood then reads:

.. math::
	\log\left[ \mbox{Likelihood} \right] =: \log\left[ P(\boldsymbol{y}|\boldsymbol{w}) \right] = \frac{N}{2}\log\left[\Phi\right]-\frac{1}{2}\Phi\,\boldsymbol{e}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{e} + \mbox{const.} \; .
	:label: eq_likl_log

Priors
======
We consider an :math:`\mathcal{MVN}` distribution for the prior of :math:`\theta` (:math:`N_{\theta}` is the length of parameter vector :math:`\theta`):

.. math::
	\theta \sim \mathcal{MVN}(\boldsymbol{m}_0, \Lambda_0^{-1}) \quad \rightarrow \quad P(\theta|\boldsymbol{m}_0,\Lambda_0)=\sqrt{ \frac{|\Lambda_0|}{(2\pi)^{N_{\theta}}} } \; e^{-0.5\,(\theta-\boldsymbol{m}_0)^T.\Lambda_0.(\theta-\boldsymbol{m}_0)} \; ,
	:label: eq_prior_theta

and a Gamma distribution for the prior of the noise parameter :math:`\Phi`:

.. math::
	\Phi \sim \Gamma(c_0,s_0) \quad \rightarrow \quad P(\Phi|c_0,s_0)=\frac{1}{\Gamma(c_0)}\frac{\Phi^{c_0-1}}{s_0^{c_0}}e^{-\frac{\Phi}{s_0}} \; .
	:label: eq_prior_Phi

One assumption in the derivations that follow is, that the *priors* of the latent parameters :math:`\theta` and :math:`\Phi` are independent, i.e.:

.. math::
	P(\boldsymbol{w}) = P(\theta, \Phi) = P(\theta)\,P(\Phi)
	:label: eq_priors_uncor

:math:`\log \left[ P(\boldsymbol{y}|\boldsymbol{w}) P(\boldsymbol{w}) \right]`
==============================================================================
We need to expand the term :math:`\log \left[ P(\boldsymbol{y}|\boldsymbol{w}) P(\boldsymbol{w}) \right]` as a required step for solving :eq:`eq_qtheta_main` and :eq:`eq_qPhi_main`. It is important to recognize, that this term is **log of the posterior** plus log of the constant **evidence**: :math:`\int P(\boldsymbol{y}|\boldsymbol{w}) P(\boldsymbol{w}) d\boldsymbol{w}`. Using :eq:`eq_likl_log`, :eq:`eq_priors_uncor`, :eq:`eq_prior_theta` and :eq:`eq_prior_Phi`, and adding all constant terms to one term, we obtain:

.. math::
	\begin{split}
	\log \left[ P(\boldsymbol{y}|\boldsymbol{w}) P(\boldsymbol{w}) \right] &= \log \left[ P(\boldsymbol{y}|\theta,\Phi) \right] + \log \left[ P(\theta) \right] + \log \left[ P(\Phi) \right]
	\\
	&=\frac{N}{2}\log\left[\Phi\right]-\frac{1}{2}\Phi\,\boldsymbol{e}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{e}
	\\
	&-\frac{1}{2}\,(\theta-\boldsymbol{m}_0)^T.\Lambda_0.(\theta-\boldsymbol{m}_0)
	\\
	&+(c_0-1) \log \left[ \Phi \right] - \frac{\Phi}{s_0} + \textcolor{blue}{\mathrm{const}\lbrace \boldsymbol{\theta},\Phi \rbrace} \, .
	\end{split}
	:label: eq_log_post

This equation can be regrouped in the following way that is more handy for the next steps.

.. math::
	\begin{split}
	\log \left[ P(\boldsymbol{y}|\boldsymbol{w}) P(\boldsymbol{w}) \right] &= -\frac{1}{2}\Phi\,\boldsymbol{e}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{e} + f_{\theta} + f_{\Phi} + \textcolor{blue}{\mathrm{const}\left\{\boldsymbol{\theta},\Phi \right\}} \quad ;
	\\
	&f_{\theta}=: -\frac{1}{2}\,(\theta-\boldsymbol{m}_0)^T.\Lambda_0.(\theta-\boldsymbol{m}_0)
	\\
	&f_{\Phi}=:(c_0-1) \log \left[ \Phi \right] - \frac{\Phi}{s_0} + \frac{N}{2}\log\left[\Phi\right] \, .
	\end{split}
	:label: eq_log_post_regrouped

The constant term that is expressed below will not influence the solution of :eq:`eq_qtheta_main` and :eq:`eq_qPhi_main`, however, it will be used for the computation of the free energy.

.. math::
    \textcolor{blue}{\mathrm{const} \lbrace \boldsymbol{\theta},\Phi \rbrace = -\frac{N+N_{\theta}}{2}\log[2\pi] -\frac{1}{2}\log[\textcolor{red}{|C_e|}] + \frac{1}{2} \log[\mathrm{det}(\Lambda_0)] + \log[1/\Gamma(c_0)]-c_0\log[s_0]} \, .

Approximated posteriors
=======================
To become able to derive suitable update equations, which will follow up, we select the approximated posteriors in exactly the same form as the priors, i.e.:

.. math::
	q_{\theta} = \mathcal{MVN}(\theta;\,\boldsymbol{m}, \Lambda^{-1}) \quad \rightarrow \quad q(\theta)=\sqrt{ \frac{|\Lambda|}{(2\pi)^N_{\theta}} } \; e^{-0.5\,(\theta-\boldsymbol{m})^T.\Lambda.(\theta-\boldsymbol{m})} \; ,
	:label: eq_post_theta

.. math::
	q_{\Phi} = \Gamma(\Phi;\,c,s) \quad \rightarrow \quad q(\Phi)=\frac{1}{\Gamma(c)}\frac{\Phi^{c-1}}{s^{c}}e^{-\frac{\Phi}{s}} \; ,
	:label: eq_post_Phi

which introduces :math:`\boldsymbol{m},\,\Lambda,\,c,\,s` as the *deterministic* latent parameters to be identified.

Taylor expansion of the model error
===================================
Another simplification made in the VB is to approximate the model error defined in :eq:`eq_e_define` by means of a first-order Taylor expansion around the posterior mean :math:`\boldsymbol{m}`:

.. math::
	\boldsymbol{e} = \boldsymbol{e}(\theta) =: \boldsymbol{y} - g(\theta) \approx \boldsymbol{k} - \boldsymbol{J} \left( \theta - \boldsymbol{m} \right) \, ,
	:label: eq_e_Taylor

with:

.. math::
	\boldsymbol{k} =: \boldsymbol{e}(\boldsymbol{m}) = \boldsymbol{y} - g(\boldsymbol{m}) \, ,
	:label: eq_k_define

and

.. math::
	\boldsymbol{J} =: \frac{dg(\theta)}{d\theta}|_{\theta=\boldsymbol{m}} = -\frac{d\boldsymbol{e}(\theta)}{d\theta}|_{\theta=\boldsymbol{m}}\, .
	:label: eq_Jk_define

Update equations
================

Update of :math:`\theta`
------------------------
Considering :eq:`eq_log_post_regrouped` and :eq:`eq_post_theta`, we start with separately expanding both sides of :eq:`eq_qtheta_main`.

.. math::
	\begin{split}
	\log \left[ q_{\theta} \right] &= -\frac{1}{2}\,(\theta-\boldsymbol{m})^T.\Lambda.(\theta-\boldsymbol{m}) + \textcolor{blue}{\mathrm{const}\left\{\boldsymbol{\theta} \right\}}
	\\
	&=-\frac{1}{2} \left( \theta^T\Lambda\theta + \theta^T\Lambda\boldsymbol{m} + \boldsymbol{m}^T\Lambda\theta \right)  + \mathrm{const_2}\left\{\boldsymbol{\theta} \right\}
	\\
	&; \quad \textcolor{blue}{\mathrm{const}\left\{\boldsymbol{\theta} \right\} = \frac{1}{2}\log\left[\,|\Lambda|\,\right] -\frac{N_{\theta}}{2}\log[2\pi]}
	\, .
	\end{split}
	:label: eq_qtheta_lhs

.. math::
	\begin{split}
	\int q_{\Phi} \log \left[ P(\boldsymbol{y}|\boldsymbol{w}) P(\boldsymbol{w}) \right] d\Phi &= -\frac{1}{2}\boldsymbol{e}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{e} \underbrace{\int \Phi q_{\Phi} d\Phi}_{=sc} + f_{\theta} \underbrace{\int q_{\Phi} d\Phi}_{=1} + \int \left( f_{\Phi} + \textcolor{blue}{\mathrm{const}\left\{\boldsymbol{\theta},\Phi \right\}} \right) q_{\Phi} d\Phi
	\\
	&= -\frac{1}{2}sc\,\boldsymbol{e}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{e} + f_{\theta} + \mbox{const}\left\{\theta\right\} \, ,
	\end{split}
	:label: eq_qtheta_rhs_0
	
where the two identities :math:`\int \Phi q_{\Phi} d\Phi=sc` and :math:`\int q_{\Phi} d\Phi=1` have been used. Equation :eq:`eq_qtheta_rhs_0` can be further simplified according to :eq:`eq_e_Taylor` and :math:`f_{\theta}` defined in :eq:`eq_log_post_regrouped`:

.. math::
	\begin{split}
	\int q_{\Phi} \log \left[ P(\boldsymbol{y}|\boldsymbol{w}) P(\boldsymbol{w}) \right] d\Phi =&-\frac{1}{2}sc\,\left( \boldsymbol{k} - \boldsymbol{J} \left( \theta - \boldsymbol{m} \right) \right) ^T.\textcolor{red}{C_e^{-1}}. \left( \boldsymbol{k} - \boldsymbol{J} \left( \theta - \boldsymbol{m} \right) \right)
	\\
	&-\frac{1}{2}\,(\theta-\boldsymbol{m}_0)^T.\Lambda_0.(\theta-\boldsymbol{m}_0) + \mbox{const}\left\{\theta\right\} \, .
	\\
	=& -\frac{1}{2} \left( \theta^T.h_1.\theta - \theta^T.h_2 - h_3.\theta \right) + \mbox{const}\left\{\theta\right\} \quad ;
	\\
	h_1 =& \Lambda_0+sc\,\boldsymbol{J}^T\textcolor{red}{C_e^{-1}}\boldsymbol{J}
	\\
	h_2=& \Lambda_0\boldsymbol{m}_0 + sc\boldsymbol{J}^T\textcolor{red}{C_e^{-1}} \left( \boldsymbol{k} + \boldsymbol{J}\boldsymbol{m} \right)
	\\
	h_3=& \boldsymbol{m}_0^T\Lambda_0 + sc \left( \boldsymbol{k}^T+\boldsymbol{m}^T\boldsymbol{J}^T \right) \textcolor{red}{C_e^{-1}}\boldsymbol{J}
	\end{split}
	:label: eq_qtheta_rhs

We assume that the matrices :math:`\Lambda_0` and :math:`\textcolor{red}{C_e^{-1}}` are symmetric, i.e. :math:`h_2^T=h_3`. Then, by comparing both sides of :eq:`eq_qtheta_main` from :eq:`eq_qtheta_lhs` and :eq:`eq_qtheta_rhs`, we obtain the following equations for the parameters (:math:`\boldsymbol{m}, \,\Lambda`) of the posterior :math:`\theta`.

.. math::
	\boxed{
	\begin{split}
	\Lambda =& \Lambda_0+sc\,\boldsymbol{J}^T\textcolor{red}{C_e^{-1}}\boldsymbol{J}
	\\
	\Lambda\boldsymbol{m} =& \Lambda_0\boldsymbol{m}_0 + sc\boldsymbol{J}^T\textcolor{red}{C_e^{-1}} \left( \boldsymbol{k} + \boldsymbol{J}\boldsymbol{m} \right)
	\end{split}
	}
	:label: eq_update_theta
	
Update of :math:`\Phi`
----------------------
A similar procedure is performed for expanding both sides of :eq:`eq_qPhi_main` using :eq:`eq_log_post_regrouped`, :eq:`eq_post_Phi` and :eq:`eq_e_Taylor`.

.. math::
	\begin{split}
	\log \left[ q_{\Phi} \right] &= \log \left[ \frac{1}{\Gamma(c)}\frac{\Phi^{c-1}}{s^{c}}e^{-\frac{\Phi}{s}} \right] = -\log\left[ \Gamma(c)\right] -c\log\left[s\right] + (c-1)\log\left[ \Phi \right]\ -\frac{\Phi}{s}
	\\
	&= (c-1)\log\left[ \Phi \right]\ -\frac{\Phi}{s} + \textcolor{blue}{\mathrm{const}\left\{\Phi \right\}}
	\\
	&; \quad \textcolor{blue}{\mathrm{const}\left\{\Phi \right\} = -\log\left[ \Gamma(c)\right] -c\log\left[s\right]}
	\; .
	\end{split}
	:label: eq_qPhi_lhs

.. math::
	\begin{split}
	\int q_{\theta} \log \left[ P(\boldsymbol{y}|\boldsymbol{w}) P(\boldsymbol{w}) \right] d\theta =& -\frac{1}{2}\Phi \int \boldsymbol{e}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{e} \, q_{\theta} d\theta + f_{\Phi} \underbrace{\int q_{\theta} d\theta}_{=1} + \int \left( f_{\theta} + \textcolor{blue}{\mathrm{const}\left\{\boldsymbol{\theta},\Phi \right\}} \right) q_{\theta} d\theta \, .
	\\
	=& -\frac{1}{2}\Phi \int \left ( \boldsymbol{k} - \boldsymbol{J} \left( \theta - \boldsymbol{m} \right) \right) ^T.\textcolor{red}{C_e^{-1}}. \left( \boldsymbol{k} - \boldsymbol{J} \left( \theta - \boldsymbol{m} \right) \right) \, q_{\theta} d\theta + f_{\Phi} + \mbox{const}\left\{\Phi\right\}
	\\
	=& -\frac{1}{2}\Phi \left( \boldsymbol{k}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{k} \underbrace{\int q_{\theta} d\theta}_{=1} + \int \left( \theta - \boldsymbol{m} \right)^T.\left( \boldsymbol{J}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{J} \right).\left( \theta - \boldsymbol{m} \right) q_{\theta} d\theta \right)
	\\
	& -\frac{1}{2}\Phi \left(  -\boldsymbol{k}^T.\textcolor{red}{C_e^{-1}}.J. \cancelto{0}{\int \left( \theta - \boldsymbol{m} \right) q_{\theta} d\theta} - \cancelto{0}{\int \left( \theta - \boldsymbol{m} \right)^T q_{\theta} d\theta} \, .\boldsymbol{J}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{k}  \right)
	\\
	&+ (c_0-1) \log \left[ \Phi \right] - \frac{\Phi}{s_0} + \frac{N}{2}\log\left[\Phi\right] + \mbox{const}\left\{\Phi\right\}
	\\
	=& -\frac{1}{2}\Phi \left( \boldsymbol{k}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{k} +\mbox{tr}\left( \Lambda^{-1}\boldsymbol{J}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{J} \right) \right) - \frac{\Phi}{s_0}
	\\
	&+ \log\left[\Phi\right]\left(c_0-1+\frac{N}{2}\right) + \mbox{const}\left\{\Phi\right\}
	\, .
	\end{split}
	:label: eq_qPhi_rhs

Note that, in the last step of :eq:`eq_qPhi_rhs` the following identity (trace trick) has been used in view of :math:`q_{\theta}` defined in :eq:`eq_post_theta`.

.. math::
	\begin{split}
	\boldsymbol{E}_{\mathcal{MVN}(\theta;\,\boldsymbol{m}, \Lambda^{-1})}\left( \left( \theta - \boldsymbol{m} \right)^TA\left( \theta - \boldsymbol{m} \right) \right) &=: \int \left( \theta - \boldsymbol{m} \right)^TA\left( \theta - \boldsymbol{m} \right) \mathcal{MVN}(\theta;\,\boldsymbol{m}, \Lambda^{-1}) d\theta
	\\
	&= \mbox{tr}\left( \Lambda^{-1}A\right) \quad ; \; \mbox{for any constant matrix } A 
	\end{split}
	:label: eq_trace_trick

Comparing both sides of :eq:`eq_qPhi_main` simplified in :eq:`eq_qPhi_lhs` and :eq:`eq_qPhi_rhs`, we obtain the following equations for the posterior noise parameters (:math:`c, \,s`).

.. math::
	\boxed{
	\begin{split}
	c =& c_0+\frac{N}{2}
	\\
	\frac{1}{s} =& \frac{1}{s_0} + \frac{1}{2}\left ( \boldsymbol{k}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{k} +\mbox{tr}\left( \Lambda^{-1}\boldsymbol{J}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{J} \right) \right)
	\end{split}
	}
	:label: eq_update_Phi

Simplification of the free energy
=================================
We look into maximizing the free energy defined in :eq:`eq_free_energy` via solving the set of algebraic equations obtained in the previous section. This optimization problem is likely to be ill-posed for highly nonlinear forward simulators (:math:`g`); i.e. to have several local maximums. To cope with this situation, it is useful to compute and keep monitoring the free energy. Considering :math:`q(\boldsymbol{w})=q_{\theta}\,q_{\Phi}` and :math:`d\boldsymbol{w}=d\theta\,d\Phi`, we start expanding :eq:`eq_free_energy`:

.. math::
	\begin{split}
	F =& \int q_{\theta}\,q_{\Phi} \log \left[ \frac{P(\boldsymbol{y}|\boldsymbol{w})\,P(\boldsymbol{w})}{q_{\theta}\,q_{\Phi}} \right] d\theta\,d\Phi
	\\
	=& \underbrace{\int q_{\theta}\,q_{\Phi} \log \left[ P(\boldsymbol{y}|\boldsymbol{w})\,P(\boldsymbol{w}) \right] d\theta\,d\Phi}_{=:F_1} - \underbrace{\int q_{\theta}\,q_{\Phi} \log \left[ q_{\theta} \right] d\theta\,d\Phi}_{=:F_2} - \underbrace{\int q_{\theta}\,q_{\Phi} \log \left[ q_{\Phi} \right] d\theta\,d\Phi}_{=:F_3} \,.
	\end{split}
	:label: eq_F_base

The expansion of the terms :math:`F_1`, :math:`F_2`, :math:`F_3` follow.

.. math::
	\begin{split}
	F_1 =& \int q_{\theta}\,q_{\Phi} \log \left[ P(\boldsymbol{y}|\boldsymbol{w})\,P(\boldsymbol{w}) \right] d\theta\,d\Phi
	\\
	=& \int q_{\theta}\,q_{\Phi} \left ( -\frac{1}{2}\Phi\,\boldsymbol{e}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{e} + f_{\theta} + f_{\Phi} + \textcolor{blue}{\mathrm{const}\left\{\boldsymbol{\theta},\Phi \right\}} \right) d\theta\,d\Phi
	\\
	=& -\frac{1}{2} \int \boldsymbol{e}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{e} \, q_{\theta}d\theta \underbrace{\int \Phi\,q_{\Phi}d\Phi}_{=sc} -\frac{1}{2} \int \left( (\theta-\boldsymbol{m}_0)^T.\Lambda_0.(\theta-\boldsymbol{m}_0) \right) q_{\theta}d\theta \underbrace{\int q_{\Phi}d\Phi}_{=1}
	\\
	& + \underbrace{\int q_{\theta}d\theta}_{=1} \int \left( (c_0-1) \log \left[ \Phi \right] - \frac{\Phi}{s_0} + \frac{N}{2}\log\left[\Phi\right] \right)d\Phi + \textcolor{blue}{\mathrm{const}\left\{\boldsymbol{\theta},\Phi \right\}}
	\\
	=& -\frac{1}{2} sc \left( \boldsymbol{k}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{k} +\mbox{tr}\left( \Lambda^{-1}\boldsymbol{J}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{J} \right) \right) -\frac{sc}{s_0} +\left( \frac{N}{2} + c_0 -1 \right) \left( \log
	\left[s\right] + \psi(c) \right)
	\\
	& -\frac{1}{2}\left( (\boldsymbol{m}-\boldsymbol{m}_0)^T.\Lambda_0.(\boldsymbol{m}-\boldsymbol{m}_0) + \mbox{tr}\left( \Lambda^{-1}\Lambda_0 \right) \right) + \textcolor{blue}{\mathrm{const}\left\{\boldsymbol{\theta},\Phi \right\}} \, .
	\end{split}
	:label: eq_F1

The first term in the above simplified expression has been rewritten according to the simplication already done through the derivation of :eq:`eq_qPhi_rhs`. We have furthermore applied the identity :math:`\int \log\left[ \Phi\right] d\Phi=\log\left[s\right] + \psi(c)` with :math:`\psi` being the di-gamma function defined as :math:`\psi(x)=:\frac{d}{dx}\ln\Gamma(x)=\frac{\Gamma'(x)}{\Gamma(x)}`, as well as the following identity that is similar to :eq:`eq_trace_trick`.

.. math::
	\begin{split}
	\boldsymbol{E}_{\mathcal{MVN}(\theta;\,\boldsymbol{m}, \Lambda^{-1})}\left[ \left( \theta - \boldsymbol{m}_0 \right)^TA\left( \theta - \boldsymbol{m}_0 \right) \right] &=: \int \left( \theta - \boldsymbol{m}_0 \right)^TA\left( \theta - \boldsymbol{m}_0 \right) \mathcal{MVN}(\theta;\,\boldsymbol{m}, \Lambda^{-1}) d\theta
	\\
	&= (\boldsymbol{m}-\boldsymbol{m}_0)^TA(\boldsymbol{m}-\boldsymbol{m}_0) + \mbox{tr}\left( \Lambda^{-1}A\right)
	\\
	& \quad ; \; \mbox{for any constant matrix } A 
	\end{split}
	:label: eq_trace_trick_2

We continue with expanding :math:`F_2` and :math:`F_3` by the help of :eq:`eq_qtheta_lhs`, :eq:`eq_trace_trick` and :eq:`eq_qPhi_lhs`:

.. math::
	\begin{split}
	F_2 &= \int q_{\theta}\,q_{\Phi} \log \left[ q_{\theta} \right] d\theta\,d\Phi = \int \log \left[ q_{\theta} \right] q_{\theta}d\theta \int q_{\Phi} d\Phi
	\\
	&= \int \left( -\frac{1}{2}\,(\theta-\boldsymbol{m})^T.\Lambda.(\theta-\boldsymbol{m}) + \textcolor{blue}{\mathrm{const}\left\{\boldsymbol{\theta} \right\}} \right) q_{\theta}d\theta
	\\
	&= -\frac{1}{2}\mathrm{tr}(\Lambda^{-1}\Lambda) +\textcolor{blue}{\mathrm{const}\left\{\boldsymbol{\theta} \right\}} =  -\frac{N_{\theta}}{2} + \textcolor{blue}{\frac{1}{2}\log\left[\,|\Lambda|\,\right] -\frac{N_{\theta}}{2}\log[2\pi]}
	\,.
	\end{split}
	:label: eq_F2

.. math::
	\begin{split}
	F_3 &= \int q_{\theta}\,q_{\Phi} \log \left[ q_{\Phi} \right] d\theta\,d\Phi = \int q_{\theta} d\theta \int \log \left[ q_{\Phi} \right] q_{\Phi}d\Phi
	\\
	&= \int \left(  (c-1)\log\left[ \Phi \right]\ -\frac{\Phi}{s} + \textcolor{blue}{\mathrm{const}\left\{\Phi \right\}} \right) q_{\Phi}d\Phi
	\\
	&= (c-1)\left( \log
	\left[s\right] + \psi(c) \right) - c \textcolor{blue}{-\log\left[ \Gamma(c)\right] -c\log\left[s\right]}
	\,.
	\end{split}
	:label: eq_F3

Substituting :eq:`eq_F1`, :eq:`eq_F2` and :eq:`eq_F3` into :eq:`eq_F_base` results in:

.. math::
	\boxed{
	\begin{split}
	F =& -\frac{1}{2} sc \left( \boldsymbol{k}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{k} +\mbox{tr}\left( \Lambda^{-1}\boldsymbol{J}^T.\textcolor{red}{C_e^{-1}}.\boldsymbol{J} \right) \right) -\frac{sc}{s_0} -\frac{1}{2}\left( (\boldsymbol{m}-\boldsymbol{m}_0)^T.\Lambda_0.(\boldsymbol{m}-\boldsymbol{m}_0) + \mbox{tr}\left( \Lambda^{-1}\Lambda_0 \right) \right)
	\\
	& + \cancelto{0}{\left( \frac{N}{2} + c_0 -c \right)} \left( \log
	\left[s\right] + \psi(c) \right) + c + \frac{N_{\theta}}{2}
	\textcolor{blue}{
		- \frac{1}{2}\log\left[\,|\Lambda|\,\right] + \cancel{\frac{N_{\theta}}{2}\log[2\pi]}
		+ \log\left[ \Gamma(c)\right] + c\log\left[s\right]
	}
	\\
	& \textcolor{blue}{
		-\frac{N+\cancel{N_{\theta}}}{2}\log[2\pi] -\frac{1}{2}\log[\textcolor{red}{|C_e|}] + \frac{1}{2} \log[\mathrm{det}(\Lambda_0)] + \log[1/\Gamma(c_0)]-c_0\log[s_0]
	}
	\, .
	\end{split}
	}
	:label: eq_F

Notice the vanishing term provided that the noise parameter :math:`c` is already updated by the first identity in :eq:`eq_update_Phi`.

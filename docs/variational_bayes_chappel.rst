Derivation of variational Bayesian for nonlinear models
================================================================
The derivation is extracted from two papers

*   Chappell, M. A.; Groves, A. R.; Whitcher, B. \& Woolrich, M. W.
    Variational Bayesian Inference for a Nonlinear Forward Model,
    IEEE Transactions on Signal Processing, 2009, 57, 223-236
    https://ieeexplore.ieee.org/document/4625948
*   Chappell, M. A.; Groves, A. R. \& Woolrich, M. W.
    The FMRIB Variational Bayes Tutorial: Variational Bayesian inference for non-linear forward model,
    https://users.fmrib.ox.ac.uk/~chappell/papers/TR07MC1.pdf

Maximization of free energy
---------------------------
Free energy:

.. math::
    \require{cancel}
    F &= \int q(\boldsymbol{w})
    \log \,\frac{P(\boldsymbol{y}|\boldsymbol{w})\,P(\boldsymbol{w})}{q(\boldsymbol{w})} d\boldsymbol{w}.\\

Mean field approximation

.. math::
    q(\boldsymbol{w}) &= \prod_{i=1}^m q_{i}(\boldsymbol{w}_i).\\

In the special case of separating between model parameters :math:`\boldsymbol{\theta}` and noise parameters
:math:`\boldsymbol{\Phi}`, the posterior is approximated by the product of two distributions:

.. math::
    q(\boldsymbol{w}) &=q_\theta q_\Phi.\\

Variational inference tries to find the approximate posterior distributions :math:`q_i(w_i)` with :math:`i\epsilon
\left\{\theta, \Phi\right\}` that maximize the free energy :math:`F(q)`. Rewriting the free energy with the mean field
approximation results in:

.. math::
    F &= \int q_{i} \, q_{\cancel{i}} \,
    \log\left[P(\boldsymbol{y}|\boldsymbol{w})\, P(\boldsymbol{w})\right]
    - q_{i} \,q_{\cancel{i}} \, \log[q_{i}]
    - q_{i} \, q_{\cancel{i}} \, \log[q_{\cancel{i}}]
    \;d \boldsymbol{w}.\\

where :math:`q_\cancel{i}` represents the product of all distributions apart from :math:`q_i`.

The function :math:`F=\int f\left(\boldsymbol{w}, q(\boldsymbol{w})\right) \;d\boldsymbol{w}` with :math:`f` given as
the integrand of the the above equations is maximized with respect to a subset of the parameters :math:`\boldsymbol{w}_i`, thus the
function is written in terms of these parameters alone.

.. math::
    F &= \int g\left(\boldsymbol{w}_i, q_{i}(\boldsymbol{w}_i)\right) \;d\boldsymbol{w}_i.\\

with

.. math::
    g\left(\boldsymbol{w}_i, q_{i}(\boldsymbol{w}_i)\right) &=
    \int f\left(\boldsymbol{w}, q(\boldsymbol{w})\right) \;d\boldsymbol{w}_\cancel{i}.\\

From variational calculus, the maximum of F is the solution of the
`Euler-Lagrange equation <https://en.wikipedia.org/wiki/Calculus_of_variations#Euler%E2%80%93Lagrange_equation>`_.

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

Moving the second term to the other side of the equation and realizing that the latter two terms do not depend
on :math:`q_{i}`

.. math::
    \log[q_i] & = \int q_{\cancel{i}}\log[P(\boldsymbol{y}|\boldsymbol{w}) P(\boldsymbol{w})]
    d\boldsymbol{w}_{\!\cancel{i}} + \mathrm{const} \\
    \log[q_{i}] & \propto \int q_{\cancel{i}}\log[P(\boldsymbol{y}|\boldsymbol{w}) P(\boldsymbol{w})]
    d\boldsymbol{w}_{\!\cancel{i}}

which is identical to eq.(5) in Chappels paper.

Log Posterior
-------------
The log-posterior :math:`L` in the above integrand is given using Bayes theorem and the assumption of parameter
and noise prior being
uncorrelated by:

.. math::
    L = & \;\log[P(\boldsymbol{y}|\boldsymbol{\theta},\Phi] +\log[P(\boldsymbol{\theta})] +\log[P(\Phi)] +
    \mathrm{const}
    \lbrace \boldsymbol{\theta},\Phi \rbrace\\[3mm]
    = & \;\log[P(\boldsymbol{y}|\boldsymbol{\theta},\Phi]+\log[\mathcal{N}(\boldsymbol{\theta};\boldsymbol{m_0},
    \Lambda_0^{-1})]+\log[\Gamma(\Phi;s_0,c_0)] + \mathrm{const}\lbrace \boldsymbol{\theta},\Phi
    \rbrace\\[2mm]
    = &  \left(-\frac{N}{2}\log[2\pi]\right) + \frac{N}{2}\log[\Phi] - \frac{1}{2} \Phi
    \boldsymbol{k}^T\boldsymbol{k} \\
    & + (\frac{1}{2}\log[2\pi^p \, \mathrm{det}(\Lambda_0^{-1})]) -\frac{1}{2} (\boldsymbol{\theta}-\boldsymbol{m}_0)^T
    \, \Lambda_0 \,(\boldsymbol{\theta}-\boldsymbol{m}_0) \\
    & + (\log[1/\Gamma(c_0)]-c_0\log[s_0]) + (c_0-1)\log[\Phi] -\frac{1}{s_0} \Phi \\[2mm]
    & + \mathrm{const} \lbrace \boldsymbol{\theta},\Phi \rbrace.\\

The constant term is caused by ignoring the evidence term :math:`P(\boldsymbol{w}|\boldsymbol{y})=\frac{P
(\boldsymbol{y}|\boldsymbol{w})\,P(\boldsymbol{w})}{P(\boldsymbol{y})}`.

Adding all terms not dependent on :math:`\boldsymbol{\theta}` and :math:`\boldsymbol{\Phi}`
[marked with ()] to the constant gives

.. math::
    L = \frac{N}{2}\log[\Phi] - \frac{1}{2} \Phi \boldsymbol{k}^T\boldsymbol{k} -\frac{1}{2}
    (\boldsymbol{\theta}-\boldsymbol{m}_0)^T \, \Lambda_0 \,(\boldsymbol{\theta}-\boldsymbol{m}_0)  + (c_0-1)
    \log[\Phi] -\frac{1}{s_0} \Phi + \mathrm{const} \lbrace \boldsymbol{\theta},\Phi \rbrace

similar to eq.(16) in Chappels paper.

Update equations
----------------
Substituting :math:`L` into the update equations results in the update equations:

.. math::
    \log[q_{\theta}] & \propto &  \int q_{\Phi} L \, d\Phi  \\
    \log[\mathcal{N}(\boldsymbol{\theta};\boldsymbol{m},\Lambda^{-1})] & \propto & \int L \, \Gamma(\Phi;s,c)
    \, d\Phi

.. math::
    \log[q_{\Phi}] & \propto &  \int q_{\theta} L \, d\boldsymbol{\theta}  \\
    \log[\mathrm{\Gamma}(\Phi;s,c)] & \propto & \int L \, \mathcal{N}(\boldsymbol{\theta};\boldsymbol{m},
    \Lambda^{-1})\,d\boldsymbol{\theta}

Update equations for parameters :math:`\boldsymbol{\theta}`
___________________________________________________________
Left hand side of the equation:

.. math::
    \log[q_{\theta}]  = &\log[\mathcal{N}(\boldsymbol{\theta};\boldsymbol{m},\Lambda^{-1})] \\
    = & -\frac{1}{2} (\boldsymbol{\theta}-\boldsymbol{m})^T \, \Lambda \,(\boldsymbol{\theta}-\boldsymbol{m}) +
    \mathrm{const}\lbrace \boldsymbol{\theta} \rbrace \\
    = &  -\frac{1}{2} [\boldsymbol{\theta}^T \Lambda \boldsymbol{\theta} - \boldsymbol{\theta}^T \Lambda
    \boldsymbol{m} - \boldsymbol{m}^T \Lambda \boldsymbol{\theta}+ \boldsymbol{m}^T \Lambda \boldsymbol{m} ]  +
    \mathrm{const}\lbrace \boldsymbol{\theta} \rbrace \\
    = & -\frac{1}{2} [\boldsymbol{\theta}^T \Lambda \boldsymbol{\theta} - \boldsymbol{\theta}^T \Lambda
    \boldsymbol{m} - \boldsymbol{m}^T \Lambda \boldsymbol{\theta}]  + \mathrm{const}\lbrace \boldsymbol{\theta}
    \rbrace\\

similar to eq.(B2) in Chappell.

.. math::
    \int q_{\Phi} L \, d\Phi  = & \int L \, \Gamma(\Phi;s,c) \, d\Phi \\
    = & -\frac{1}{2} \boldsymbol{k}^T\boldsymbol{k} \int \Phi \, \Gamma(\Phi;s,c) \, d\Phi -\frac{1}{2}
    (\boldsymbol{\theta}-\boldsymbol{m}_0)^T \, \Lambda_0 \,(\boldsymbol{\theta}-\boldsymbol{m}_0) \int \Gamma(\Phi;s,c) \,
    d\Phi \\
    &  +  \int \mathrm{const}\lbrace \boldsymbol{\theta} \rbrace(\Phi) \, \Gamma(\Phi;s,c) \, d\Phi \\
    = & -\frac{1}{2} \boldsymbol{k}^T\boldsymbol{k} \, sc -\frac{1}{2}  (\boldsymbol{\theta}-\boldsymbol{m}_0)^T \,
    \Lambda_0 \,(\boldsymbol{\theta}-\boldsymbol{m}_0)
    + \mathrm{const}\lbrace \boldsymbol{\theta} \rbrace,\\

where a Taylor expansion in :math:`\boldsymbol{k}` can be used:

.. math::
    \boldsymbol{k}(\boldsymbol{\theta}) \approx \boldsymbol{k}(\boldsymbol{m}) + \boldsymbol{J}_k \,
    (\boldsymbol{\theta}-\boldsymbol{m}).

This results in:

.. math::
    = & -\frac{1}{2} (\boldsymbol{k}(\boldsymbol{m}) + \boldsymbol{J}_k \, (\boldsymbol{\theta}-\boldsymbol{m}))^T
    (\boldsymbol{k}(\boldsymbol{m}) + \boldsymbol{J}_k \, (\boldsymbol{\theta}-\boldsymbol{m})) \, sc
    -\frac{1}{2}(\boldsymbol{\theta}-\boldsymbol{m}_0)^T \, \Lambda_0 \,(\boldsymbol{\theta}-\boldsymbol{m}_0)\\
    &+\mathrm{const}\lbrace \boldsymbol{\theta} \rbrace \\
    = & -\frac{1}{2} [\boldsymbol{\theta}^T (\Lambda_0 + sc\,\boldsymbol{J}_k^T \boldsymbol{J}_k) \boldsymbol{\theta}
    - \boldsymbol{\theta}^T (\Lambda_0 \boldsymbol{m}_0 + sc \, \boldsymbol{J}_k^T\boldsymbol{k}_m - sc\,
    \boldsymbol{J}_k^T \boldsymbol{J}_k \boldsymbol{m}) \\
    &- (\boldsymbol{m}_0^T \Lambda_0 + sc\,\boldsymbol{k}(m)^T\boldsymbol{J}_k
    - sc\, \boldsymbol{m}^T\boldsymbol{J}_k^T\boldsymbol{J}_k) \boldsymbol{\theta}] + \mathrm{const}\lbrace \boldsymbol{\theta} \rbrace\\
    = & -\frac{1}{2} [\boldsymbol{\theta}^T (\Lambda_0 + sc\,\boldsymbol{J}_k^T \boldsymbol{J}_k) \boldsymbol{\theta}
    - \boldsymbol{\theta}^T (\Lambda_0 \boldsymbol{m}_0 + sc \, \boldsymbol{J}_k^T(\boldsymbol{k}_m -\boldsymbol{J}_k \boldsymbol{m}))\\
    &- (\boldsymbol{m}_0^T \Lambda_0 + sc\,(\boldsymbol{k}_m^T - \boldsymbol{m}^T\boldsymbol{J}_k^T)
    \boldsymbol{J}_k) \boldsymbol{\theta}]
    + \mathrm{const}\lbrace \boldsymbol{\theta} \rbrace.

Compare to the left hand side while omitting the terms constant in :math:`\boldsymbol{\theta}` gives:

.. math::
    -\frac{1}{2} [\boldsymbol{\theta}^T \Lambda \boldsymbol{\theta} - \boldsymbol{\theta}^T \Lambda \boldsymbol{m} - \boldsymbol{m}^T \Lambda \boldsymbol{\theta}]
    & \propto & -\frac{1}{2} [\boldsymbol{\theta}^T (\Lambda_0 + sc\,\boldsymbol{J}_k^T \boldsymbol{J}_k) \boldsymbol{\theta} \\
    & & - \boldsymbol{\theta}^T (\Lambda_0 \boldsymbol{m}_0 + sc \, \boldsymbol{J}_k^T(\boldsymbol{k}_m - \boldsymbol{J}_k \boldsymbol{m})) \\
    & & - (\boldsymbol{m}_0^T \Lambda_0 + sc\,(\boldsymbol{k}_m^T - \boldsymbol{m}^T\boldsymbol{J}_k^T)
    \boldsymbol{J}_k) \boldsymbol{\theta}].

resulting in the update equations

.. math::
    \Lambda & =& \Lambda_0 +  sc\,\boldsymbol{J}_k^T \boldsymbol{J}_k \\
    \Lambda \boldsymbol{m} &=& \Lambda_0 \boldsymbol{m}_0 + sc \, \boldsymbol{J}_k^T(\boldsymbol{k}_m -
    \boldsymbol{J}_k \boldsymbol{m}).

similar to Chappell eq. 19/20 with :math:`\boldsymbol{J}=-\boldsymbol{J}_k` (no iteration required)

Update equations noise :math:`\Phi`
___________________________________
Left hand side

.. math::
    \log[q_{\Phi}] & = &\log[\Gamma(\Phi;s,c)] \\
    & = & (c-1)\log[\Phi] + \frac{\Phi}{s} + \mathrm{const} \lbrace \Phi \rbrace\\

see Chappell eq.(B9).

.. math::
    \int q_{\theta} L \, d\boldsymbol{\theta}  = & \int L \, \mathcal{N}(\boldsymbol{\theta};\boldsymbol{m},
    \Lambda^{-1})\, d\boldsymbol{\theta} \\
    = & -\frac{1}{2} \Phi \int  \boldsymbol{k}^T \boldsymbol{k} \mathcal{N}(\boldsymbol{\theta};\boldsymbol{m},
    \Lambda^{-1})\, d\boldsymbol{\theta} \\
    & + ( \frac{N}{2}\log[\Phi] + (c_0-1)\log[\Phi]-\frac{\Phi}{s_0} )\int \mathcal{N}(\boldsymbol{\theta};
    \boldsymbol{m},\Lambda^{-1})\, d\boldsymbol{\theta} \\
    & + \mathrm{const}\lbrace \boldsymbol{\Phi} \rbrace\\


use Taylor expansion and eq B12 Chappell, :math:`(\boldsymbol{\theta}-\boldsymbol{m})`-terms integrate to zero.

.. math::
    \int  \boldsymbol{k}^T \boldsymbol{k} \mathcal{N}(\boldsymbol{\theta};\boldsymbol{m},\Lambda^{-1})\,
    d\boldsymbol{\theta}
    = & \int (\boldsymbol{k}_m + \boldsymbol{J}_k \, (\boldsymbol{\theta}- \boldsymbol{m}))^T (\boldsymbol{k}_m +
    \boldsymbol{J}_k \, (\boldsymbol{\theta}- \boldsymbol{m})) \mathcal{N}(\boldsymbol{\theta};\boldsymbol{m},
    \Lambda^{-1})\, d\boldsymbol{\theta} \\
    = &\, \boldsymbol{k}_m^T \boldsymbol{k}_m \int \mathcal{N}(\boldsymbol{\theta};\boldsymbol{m},\Lambda^{-1})\,
    d\boldsymbol{\theta} \\
    & + \int \cancel{(\boldsymbol{k}_m^T\boldsymbol{J}_k(\boldsymbol{\theta}-\boldsymbol{m})} + \cancel{
    (\boldsymbol{J}_k(\boldsymbol{\theta}-\boldsymbol{m}))^T\boldsymbol{k}_m )}  \mathcal{N}(\boldsymbol{\theta};
    \boldsymbol{m},\Lambda^{-1})\, d\boldsymbol{\theta} \\
    & + \underbrace{\int (\boldsymbol{\theta}-\boldsymbol{m})^T \boldsymbol{J}_k^T\boldsymbol{J}_k
    (\boldsymbol{\theta}-\boldsymbol{m}) \mathcal{N}(\boldsymbol{\theta};\boldsymbol{m},\Lambda^{-1})\,
    d\boldsymbol{\theta}}_{\mathrm{tr}(\Lambda^{-1}\boldsymbol{J}_k^T \boldsymbol{J}_k)}.\\

Compare to the left hand side while omitting the terms constant in :math:`\Phi` and noting that the integration over the (normal)
density function is one results in:

.. math::
    (c-1)\log[\Phi] + \frac{\Phi}{s}  \propto & \frac{N}{2}\log[\Phi] + (c_0-1)
    \log[\Phi]-\frac{\Phi}{s_0} -\frac{1}{2}\Phi(\boldsymbol{k}_m^T \boldsymbol{k}_m +
    \mathrm{tr}(\Lambda^{-1}\boldsymbol{J}_k^T \boldsymbol{J}_k)) \\
    \propto &  (\frac{N}{2}+ c_0-1 )\log[\Phi] - \Phi (\frac{1}{s_0} + \frac{1}{2}(\boldsymbol{k}_m^T 
    \boldsymbol{k}_m + \mathrm{tr}(\Lambda^{-1}\boldsymbol{J}_k^T \boldsymbol{J}_k))).\\

.. math::
    c \cancel{-1} &=& \frac{N}{2} + c_0 \cancel{-1} \\
    \frac{1}{s} &=& \frac{1}{s_0} + \frac{1}{2}(\boldsymbol{k}_m^T \boldsymbol{k}_m + \mathrm{tr}(\Lambda^{-1}\boldsymbol{J}_k^T \boldsymbol{J}_k))

similar to Chappell eq. 21/22 with :math:`\boldsymbol{J}=-\boldsymbol{J}_k`

Summary of equations to solve
-----------------------------

.. math::
    \Lambda & =& \Lambda_0 +  sc\,\boldsymbol{J}_k^T \boldsymbol{J}_k \\
    \Lambda \boldsymbol{m} &=& \Lambda_0 \boldsymbol{m}_0 + sc \, \boldsymbol{J}_k^T(\boldsymbol{k}_m - \boldsymbol{J}_k \boldsymbol{m})\\
    c &=& \frac{N}{2} + c_0  \\
    \frac{1}{s} &=& \frac{1}{s_0} + \frac{1}{2}(\boldsymbol{k}_m^T \boldsymbol{k}_m + \mathrm{tr}(\Lambda^{-1}\boldsymbol{J}_k^T \boldsymbol{J}_k))

reduces to two equations for :math:`\boldsymbol{m}` and :math:`s` by inserting eq 1 and 3 into 2 and 4

.. math::
    (\Lambda_0 +  s (\frac{N}{2} + c_0 ) \,\boldsymbol{J}_k^T \boldsymbol{J}_k)\boldsymbol{m} &=& \Lambda_0 \boldsymbol{m}_0 + s (\frac{N}{2} + c_0 ) \, \boldsymbol{J}_k^T(\boldsymbol{k}_m - \boldsymbol{J}_k \boldsymbol{m}) \Rightarrow \boldsymbol{m} = f_1(\boldsymbol{m},s)\\
    \frac{1}{s} &=& \frac{1}{s_0} + \frac{1}{2}(\boldsymbol{k}_m^T \boldsymbol{k}_m + \mathrm{tr}((\Lambda_0 +  s (\frac{N}{2} + c_0)\,\boldsymbol{J}_k^T \boldsymbol{J}_k)^{-1}\boldsymbol{J}_k^T \boldsymbol{J}_k))  \Rightarrow s = f_2(\boldsymbol{m},s)

e.g. using fixed point iteration until parameter converged.

Additional convergence check via :math:`F`
------------------------------------------
"Convergence [...] guarantee no longer holds [...]. A typical consequence is that VB algorithm cycles through a limited set of solutions without settling on asingle set of values." Chappell sec B

Monitoring free-energy for that case (**notation to be improved**)

.. math::
    F =& \int q_{\theta} \, q_{\Phi}\log \,\frac{P(\boldsymbol{y}|\boldsymbol{w})\,P(\boldsymbol{w})}{q_{\theta} \, q_{\Phi}} dw  \\
    =& \int q_{\theta} \, q_{\Phi} \,\log[P(\boldsymbol{y}|\boldsymbol{\theta},\Phi)\,P(\boldsymbol{\theta},\Phi)] - q_{\theta} \, q_{\Phi} \,\log[q_{\theta}] - q_{\theta} \, q_{\Phi} \,\log[q_{\Phi}] d\boldsymbol{\theta}d\Phi \\
    = & \int \mathcal{N}(\boldsymbol{\theta}) \Gamma(\Phi) L d\Phi d\boldsymbol{\theta} -\int \mathcal{N}(\boldsymbol{\theta}) \Gamma(\Phi)\log[\mathcal{N}(\boldsymbol{\theta})] d\Phi d\boldsymbol{\theta}\\
    & - \int \mathcal{N}(\boldsymbol{\theta}) \Gamma(\Phi)\log[\Gamma(\Phi)] d\Phi d\boldsymbol{\theta}

.. math::
    1 = &  \int \mathcal{N}(\boldsymbol{\theta}) \Gamma(\Phi) L d\Phi d\boldsymbol{\theta}   \\
    = & (\frac{N}{2}+(c_0-1)) \int\log[\Phi] \, \Gamma\, d\Phi \int \mathcal{N} \,   d\boldsymbol{\theta}
    \color{red}{ =???  (\frac{N}{2}+c_0-1)(log[s]-\psi(c))}\\
    & - \frac{1}{2} \int \Phi \boldsymbol{k}^T\boldsymbol{k} \, \Gamma\,\mathcal{N} \, d\Phi \,   d\boldsymbol{\theta} \color{red}{  =  - \frac{1}{2} \int \Phi \, \Gamma\,d\Phi \int \boldsymbol{k}^T\boldsymbol{k}\,\mathcal{N} d\boldsymbol{\theta}}  \\
    & -\frac{1}{2} \int (\boldsymbol{\theta}-\boldsymbol{m}_0)^T \Lambda_0 (\boldsymbol{\theta}-\boldsymbol{m}_0)
    \mathcal{N} \,   d\boldsymbol{\theta} \, \int \Gamma\, d\Phi \color{red}{\overbrace{=}^{eq B12, mean terms
    vanish}
    -\frac{1}{2} ((\boldsymbol{m}-\boldsymbol{m}_0)\Lambda_0(\boldsymbol{m}-\boldsymbol{m}_0)+\mathrm{tr}(\Lambda^{-1}\Lambda_0))  }  \\
    & -\frac{1}{s_0} \int \Phi \, \Gamma\ d\Phi \int \mathcal{N} \, d\boldsymbol{\theta} \color{red}{ =  - \frac{sc}{s_0}  } \\
    & + \int const \, \Gamma\, \mathcal{N} \, d\boldsymbol{\theta}\, d\Phi \color{red}{  =  const  } \\
    = &  (\frac{N}{2}+c_0-1)(log[s]-\psi(c)) - \frac{1}{2} sc (\boldsymbol{k}_m^T\boldsymbol{k}_m + \mathrm{tr}(\Lambda^{-1}\boldsymbol{J}_k^{T}\boldsymbol{J}_k)) -\frac{1}{2} ((\boldsymbol{m}-\boldsymbol{m}_0)\Lambda_0(\boldsymbol{m}-\boldsymbol{m}_0)\\
    & +\mathrm{tr}(\Lambda^{-1}\Lambda_0))  - \frac{sc}{s_0} + const

.. math::
    2 = & -\int \mathcal{N}(\boldsymbol{\theta}) \Gamma(\Phi)\log[\mathcal{N}(\boldsymbol{\theta})] d\Phi d\boldsymbol{\theta}\\
    = & - \int \Gamma \, d\Phi \, \int \mathcal{N} \,\log[\mathcal{N}] d\boldsymbol{\theta} \\
    = &??? - \int \mathcal{N} \, (const + \frac{1}{2}\log[det \Lambda^{-1}] - \frac{1}{2}(\boldsymbol{\theta} -
    \boldsymbol{m})^T \Lambda (\boldsymbol{\theta} - \boldsymbol{m}))  d\boldsymbol{\theta} \\
    = & const - \frac{1}{2}\log[det \Lambda^{-1}] + \frac{1}{2}\mathrm{tr}(\Lambda^{-1}\Lambda) \\
    = & const - \frac{1}{2}\log[det (\Lambda^{-1})] \\
    = & const - \frac{1}{2}\log[\frac{1}{det \Lambda}]\\
    = & const \cancel{- \frac{1}{2}\log[1]} + \frac{1}{2}\log[det \Lambda]

.. math::
    3 = &  - \int \mathcal{N}(\boldsymbol{\theta}) \Gamma(\Phi)\log[\Gamma(\Phi)] d\Phi d\boldsymbol{\theta}\\
    = & - \int \mathcal{N} \, d\boldsymbol{\theta} \, \int \Gamma \,\log[\Gamma] d\Phi \\
    = &??? - \int \Gamma \, (\log[1/\Gamma_c] - c\log[s] + (c-1)\log[\Phi] - \frac{\Phi}{s})  d\Phi \\
    = &  - \int \Gamma \, (\log[1/\Gamma_c] - c\log[s]) d\Phi - \int \Gamma \,((c-1)\log[\Phi] - \frac{\Phi}{s})
    d\Phi \\
    = & - (\log[1/\Gamma_c] - c\log[s]) + \frac{1}{s}\int \Phi \, \Gamma \, d\Phi - (c-1) \int \log[\Phi] \, \Gamma
    \, d\Phi \\
    = & - (\log[1/\Gamma_c] - c\log[s]) + \frac{\cancel{s}c}{\cancel{s}} - (c-1)(\log[s]+\psi(c)) \\
    = & +\log[\Gamma_c] + c\log[s]) + \frac{\cancel{s}c}{\cancel{s}} - (c-1)(\log[s]+\psi(c))

.. math::
    F =& (\frac{N}{2}+c_0-1)(\log[s]-\psi(c)) - \frac{1}{2} \color{green}{sc} (\boldsymbol{k}_m^T\boldsymbol{k}_m +
    \mathrm{tr}(\Lambda^{-1}\boldsymbol{J}_k^{T}\boldsymbol{J}_k)) \\
    & -\frac{1}{2} ((\boldsymbol{m}-\boldsymbol{m}_0)\Lambda_0(\boldsymbol{m}-\boldsymbol{m}_0) +\mathrm{tr}(\Lambda^{-1}\Lambda_0))  - \frac{sc}{s_0} \\
    & + \frac{1}{2}\log[det \Lambda] \\
    &  \color{green}{ +\log[\Gamma_c] + c\log[s] + \frac{\cancel{s}c}{\cancel{s}} - (c-1)(\log[s]+\psi(c)) }\\
    & + const

not the same as in Chappell eq 23

PLEASE CHECK it !!!









Using the free energy definition in equation.

.. math::
    :label: abc

    a &= b\\


Euler's identity, equation :eq:`abc`, was elected one of the most
beautiful mathematical formulas.

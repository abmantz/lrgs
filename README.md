This branch is currently a sandbox for implementing the model used by LRGS in [`Stan`](http://mc-stan.org), a Hamiltonian Monte Carlo sampler, to see how it compares to the conjugate Gibbs sampling implemented in [`R`](https://github.com/abmantz/lrgs/tree/R) and [`python`](https://github.com/abmantz/lrgs/tree/python).

References:
* [K07](https://arxiv.org/abs/0705.2774) especially sections 4 and 6
* [M16](https://arxiv.org/abs/1509.00908) sections 2-3


<a href="http://ascl.net/1602.005"><img src="https://img.shields.io/badge/ascl-1602.005-blue.svg?colorB=262255" alt="ascl:1602.005" /></a>

# LRGS: Linear Regression by Gibbs Sampling

Code implementing a Gibbs sampler to deal with the problem of multivariate linear regression with uncertainties in all measured quantities and intrinsic scatter. Full details can be found in [this paper](http://arxiv.org/abs/1509.00908), the abstract of which appears below.

Kelly (2007, hereafter K07) described an efficient algorithm, using Gibbs sampling, for performing linear regression in the fairly general case where non-zero measurement errors exist for both the covariates and response variables, where these measurements may be correlated (for the same data point), where the response variable is affected by intrinsic scatter in addition to measurement error, and where the prior distribution of covariates is modeled by a flexible mixture of Gaussians rather than assumed to be uniform. Here I extend the K07 algorithm in two ways. First, the procedure is generalized to the case of multiple response variables. Second, I describe how to model the prior distribution of covariates using a Dirichlet process, which can be thought of as a Gaussian mixture where the number of mixture components is learned from the data. I present an example of multivariate regression using the extended algorithm, namely fitting scaling relations of the gas mass, temperature, and luminosity of dynamically relaxed galaxy clusters as a function of their mass and redshift. An implementation of the Gibbs sampler in the R language, called LRGS, is provided. 

For questions, comments, requests, problems, etc. use the [issues](https://github.com/abmantz/lrgs/issues).

LRGS for Stan does not exist! We're just playing. (Yet there is still a [VERSION.md](VERSION.md).)

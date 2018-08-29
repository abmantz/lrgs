.. image:: https://img.shields.io/badge/ascl-1602.005-blue.svg?colorB=262255
   :alt: ascl:1602.005
   :target: http://ascl.net/1602.005
.. image:: https://img.shields.io/pypi/v/lrgs.svg
   :alt: PyPi
   :target: https://pypi.python.org/pypi/lrgs
.. image:: https://img.shields.io/pypi/l/lrgs.svg
   :alt: MIT
   :target: https://opensource.org/licenses/MIT

=====================================================================================
LRGS: Linear Regression by Gibbs Sampling
=====================================================================================

Code implementing a Gibbs sampler to deal with the problem of multivariate linear regression with uncertainties in all measured quantities and intrinsic scatter. Full details can be found in `this paper <http://arxiv.org/abs/1509.00908>`_, the abstract of which appears below. (The paper describes an implementation in the R language, while this package is a port of the method to Python.)

Kelly (2007, hereafter K07) described an efficient algorithm, using Gibbs sampling, for performing linear regression in the fairly general case where non-zero measurement errors exist for both the covariates and response variables, where these measurements may be correlated (for the same data point), where the response variable is affected by intrinsic scatter in addition to measurement error, and where the prior distribution of covariates is modeled by a flexible mixture of Gaussians rather than assumed to be uniform. Here I extend the K07 algorithm in two ways. First, the procedure is generalized to the case of multiple response variables. Second, I describe how to model the prior distribution of covariates using a Dirichlet process, which can be thought of as a Gaussian mixture where the number of mixture components is learned from the data. I present an example of multivariate regression using the extended algorithm, namely fitting scaling relations of the gas mass, temperature, and luminosity of dynamically relaxed galaxy clusters as a function of their mass and redshift. An implementation of the Gibbs sampler in the R language, called LRGS, is provided.

For questions, comments, requests, problems, etc. use the `GitHub issues <https://github.com/abmantz/lrgs/issues>`_.

Status
======
LRGS for Python is currently in alpha. It has not been fully vetted, and some features of the R version are not implemented (see `VERSION.md <https://github.com/abmantz/lrgs/blob/python/VERSION.md>`_).

Installation
============

Automatic
---------

Install from PyPI by running ``pip install lrgs``.

Manual
------

Download ``lrgs/lrgs.py`` and put it somewhere on your ``PYTHONPATH``. You will need to have the ``numpy`` and ``scipy`` packages installed.

Usage and Help
==============

Documentation is sparse at this point, but an example notebook can be found `here <https://github.com/abmantz/lrgs/tree/master/notebooks>`_.

# BayesianIntegral

| Build | Coverage | Documentation |
|-------|----------|---------------|
| [![Build status](https://github.com/s-baumann/BayesianIntegral.jl/workflows/CI/badge.svg)](https://github.com/s-baumann/BayesianIntegral.jl/actions) | [![codecov](https://codecov.io/gh/s-baumann/BayesianIntegral.jl/branch/master/graph/badge.svg?token=T8FYN8PRC5)](https://codecov.io/gh/s-baumann/BayesianIntegral.jl) | [![docs-latest-img](https://img.shields.io/badge/docs-latest-blue.svg)](https://s-baumann.github.io/BayesianIntegral.jl/dev/index.html) |

This package uses the term Bayesian Integration to mean approximating a function with a kriging metamodel (aka a gaussian process model) and then integrating under it. A kriging metamodel has the nice feature that uncertainty about the nature of the function is explicitly modelled (unlike for instance a approximation with Chebyshev polynomials) and the Bayesian Integral uses this feature to give a Gaussian distribution representing the probabilities of various integral values. The output of the bayesian_integral_gaussian_exponential function is the expectation and variance of this distribution.

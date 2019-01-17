# BayesianIntegral

This does Bayesian integration of functions of the form:

$ \int_{x \in \Re^d} f(x) g(x) $

Where d is the dimensionality of the space (so x is d dimensional), $$f(x)$$ is the function of interest and $$g(x)$$ is a pdf representing the density of each $x$ value.

By Bayesian Integration I mean approximating a function with a kriging metamodel (aka a gaussian process model) and then integrating under it. A kriging metamodel has the nice feature that uncertainty about the nature of the function is explicitly modelled (unlike for instance a Support Vector Machine) and the Bayesian Integral uses this feature to give both an expectation of the integral as well as a variance.

## Contents

```@contents
pages = ["index.md"]
Depth = 2
```

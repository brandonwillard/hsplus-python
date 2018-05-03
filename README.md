# Introduction

This Python package provides implementations for the Horseshoe and Horseshoe+ prior
calculations found in ["The Horseshoe+ Estimator of Ultra-Sparse Signals"](hsplus),
["	Default Bayesian analysis with global-local shrinkage priors"](default) and 
["Prediction risk for global-local shrinkage regression"](predict).

Specifically,

* Generic numeric and symbolic functions--via [`mpmath`](mp) and [SymPy](sp),
  respectively--for bivariate confluent hypergeometric functions.
* Expectation and moment calculations for the hypergeometric inverted-beta distribution.
* SURE values.


# Installation

Install directly from the repository with the following:
```
$ pip install git+https://github.com/brandonwillard/hsplus-python
```

## Development
After cloning the Git repository, use the `-e` option in the project directory: 
```
$ pip install -e ./
```


[mp]:http://mpmath.org/
[sp]:http://www.sympy.org/
[hsplus]:https://arxiv.org/abs/1502.00560
[predict]:https://arxiv.org/abs/1605.04796
[default]:https://arxiv.org/abs/1510.03516

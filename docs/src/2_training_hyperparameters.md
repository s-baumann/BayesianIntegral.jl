# 2.0 Training Hyperparameters

This package currently supports two ways of training hyperparameters. The first is

## 2.1 Maximum Likelihood with Stochastic Gradient Descent

This is done by the calibrate_by_ML_with_SGD function. The procedure is:
* Extract a sample of the requested size from the dataset. Sampling is done without replacement (or else the K matrix is singular and not invertible)
* Find the likelihood of the dataset given the input hyperparameters. Also find the marginal likelihood (with respect to all hyperparameters) and use the Newtonian method to suggest another set of hyperparameters. The step to the new parameterset (as chosen by the Newtonian method) can be adjusted by the step_multiple parameter.
This process continues for a user-specifiable number of iterates.

Note that using stochastic gradient descent is important here as the major time here is in inverting an $$N x N$$ matrix which has a complexity of about $$O(N^2.3)$$. Thus if only 10\% of observations are used in each iterate this makes the calibration more than 100 times faster than using all observations.

This function can be used in the following way:
```
old_cov_func_parameters = gaussian_kernel_hyperparameters(1.0, repeat([10.0] , outer = dims))
steps = 100                           # How many optimisation steps
batch_size = 5                        # Number of observations per sample
step_multiple = 1.0                   # How far to step
noise = 0.001                         # Noise parameter
seed = 2                              # Random number seed (for gathering random samples)
new_cov_func_parameters = calibrate_by_ML_with_SGD(X, y, cov_func_parameters, steps, batch_size, step_multiple, noise, seed)
```

## 2.2 RProp



This function can be used in the following way:
```
old_cov_func_parameters = gaussian_kernel_hyperparameters(1.0, repeat([10.0] , outer = dims))
params  = RProp_params(1.01,0.99,0.2,5.0,0.5)    # parameters for the RProp algorithm.
noise = 0.001                                    # Noise parameter
MaxIter = 2000                                   # Number of steps
new_cov_func_parameters = train_with_RProp(X, y, cov_func_parameters, MaxIter, noise, params)
```

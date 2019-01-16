using BayesianIntegral
using Distributions: Normal, pdf
using Sobol
using LinearAlgebra
standard_normal = Normal(0.0,1.0)
tol = 10*eps()
# Here we are going to divide the n dimensional space into 2^N cubes.
# Half of these cubes will be of value 7 and the other half have size 2.
# Each cube has equal probability. So we should have 4.5 integral.

# In ten dimensinos
samples = 50
dims = 10
p(x) = 1.0
s = SobolSeq(dims)
X = convert(Array{Float64}, hcat([next!(s, repeat([0.5] , outer = dims)     ) for i = 1:samples]...)')

function f(X)
    positives = length(X[X .> 0.5])
    if (positives % 2) == 1
        return 7.0
    else
        return 2.0
    end
end

y = Array{Float64,1}(undef, samples)
for i in 1:samples
    y[i] = f(X[i,:])
end
cov_func_parameters = gaussian_kernel_hyperparameters(1.0, repeat([1.0] , outer = dims))
prob_means = repeat([0.0] , outer = dims)
covar = Symmetric(diagm(0 => ones(dims)))
expectation_of_integral, var_of_integral = bayesian_integral_gaussian_exponential(X, y, prob_means, covar, cov_func_parameters)



steps = 20
batch_size = 50
step_multiple = 0.02
seed = 1988
noise = 0.001


marginal_likelihood, K, invK = marginal_likelihood_gaussian_derivatives(X, y, cov_func_parameters, noise)
old_likelihood = log_likelihood(y, K; invK = invK)
better_w_0, better_w_i =  calibrate_by_ML_with_SGD(X, y, w_0, w_i, steps, batch_size, step_multiple, noise, seed)
marginal_likelihood, K, invK = marginal_likelihood_gaussian_derivatives( X, y, better_w_0, better_w_i )
new_likelihood = log_likelihood(y, K, invK; invK = invK)

prob_means = repeat([0.0] , outer = dims)
covar = Symmetric(diagm(0 => ones(dims)))
bayesian_integral_gaussian_exponential(X, y, prob_means, covar, better_w_0, better_w_i, noise)
bayesian_integral_gaussian_exponential(X, y, prob_means, covar, w_0, w_i )

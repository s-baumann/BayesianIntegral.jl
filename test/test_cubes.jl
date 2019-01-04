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
    positives = length(X[X .> 0])
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

w_0 = 1.0
w_i = repeat([1.0] , outer = dims)
prob_means = repeat([0.0] , outer = dims)
covar = Symmetric(diagm(0 => ones(dims)))
cov_func_parameters = vcat(w_0,w_i)

steps = 20
batch_size = 50
step_multiple = 0.02
seed = 1988
noise = 0.001


marginal_likelihood, K, invK = marginal_likelihood_gaussian_derivatives( X, y, w_0, w_i, noise)
old_likelihood = log_likelihood(y, K, invK)
better_w_0, better_w_i =  solve_for_weights_gaussian(X, y, w_0, w_i, steps, batch_size, step_multiple, noise, seed)
marginal_likelihood, K, invK = marginal_likelihood_gaussian_derivatives( X, y, better_w_0, better_w_i )
new_likelihood = log_likelihood(y, K, invK)

prob_means = repeat([0.0] , outer = dims)
covar = Symmetric(diagm(0 => ones(dims)))
bayesian_integral_exponential( X, y , prob_means , covar, better_w_0, better_w_i, noise )
bayesian_integral_exponential( X, y , prob_means , covar, w_0, w_i )

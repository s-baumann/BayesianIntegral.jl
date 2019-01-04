using BayesianIntegral: K_matrix, K_matrix_with_marginals, gaussian_kernel, marginal_gaussian_kernel, marginal_likelihood_gaussian_derivatives, log_likelihood, solve_for_weights_gaussian
using Distributions: Normal, pdf
using Sobol
using LinearAlgebra
standard_normal = Normal(0.0,1.0)
tol = 10*eps()

# In ten dimensinos
samples = 50
dims = 10
p(x) = 1.0
s = SobolSeq(dims)
X = convert(Array{Float64}, hcat([next!(s, repeat([0.5] , outer = dims)     ) for i = 1:samples]...)')
#X = rand(samples, dims)
y = repeat([1.0] , outer = samples)
w_0 = 1.0
w_i = repeat([1.0] , outer = dims)
prob_means = repeat([0.0] , outer = dims)
covar = Symmetric(diagm(0 => ones(length(dims))))
cov_func_parameters = vcat(w_0,w_i)
noise = 0.01

cov = K_matrix(X, gaussian_kernel, cov_func_parameters, noise )
cov2, marginal_covs =  K_matrix_with_marginals(X, marginal_gaussian_kernel, cov_func_parameters, noise )
sum(abs.(cov) - abs.(cov2)) < tol
abs(sum(abs.(marginal_covs[:,:,3]) - abs.(marginal_covs[:,:,9]))) > tol

marginal_likelihood, K, invK = marginal_likelihood_gaussian_derivatives( X, y, w_0, w_i )
likelihood = log_likelihood(y, K, invK)
bump = 0.001
marginal_likelihood2, K2, invK2 = marginal_likelihood_gaussian_derivatives( X .+ bump, y, w_0, w_i )
likelihood2 = log_likelihood(y, K2, invK2)
likelihood2 - likelihood - sum(bump .* marginal_likelihood)

steps = 50
batch_size = 50
step_multiple = 0.02
seed = 1988
noise = 0.02
better_w_0, better_w_i =  solve_for_weights_gaussian(X, y, w_0, w_i, steps, batch_size, step_multiple,noise , seed)

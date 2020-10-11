

# """
#     bayesian_integral_gaussian_exponential( X::Array{Float64,2}, f::Array{Float64,1} , prob_means::Array{Float64} , covar::Hermitian{Float64,Array{Float64,2}}, w_0::Float64, w_i::Array{Float64}, noise::Float64 = 0.0)
# Returns the expectation and variance of the integral of a kriging model defined by the evaluations specified by X and f, the hyperparameters  w_0 & w_i and the noise value.
# The integration performed is: int_{x in X} f(x) p(x) dx
# Where f(x) is the function which is approximated in the kriging map by an exponential covariance function and p(x) is
# the pdf which is multivariate gaussian.
# """
# function bayesian_integral_gaussian_exponential(X::AbstractArray{T,2}, f::AbstractArray{R,1} , prob_means::AbstractArray{U,1}, covar::Hermitian, cov_func_parameters::GaussianKernelHyperparameters, noise::Real = 0.0 )  where T<:Real where R<:Real where U<:Real
#     ndim = length(cov_func_parameters.w_i)
#     nobs = size(X)[1]
#     A = diagm(0 => cov_func_parameters.w_i.^2)
#     K = K_matrix(X, BayesianIntegral.gaussian_kernel, cov_func_parameters, noise)
#     invA = inv(A)
#     invK = inv(K)
#     AplusBinv = inv(A + covar)
#
#     multipl = cov_func_parameters.w_0 * LinearAlgebra.det(invA * covar + diagm(0 => ones(ndim)))^(-0.5)
#     z = Array{promote_type(T,U,typeof(AplusBinv),typeof(multipl)),1}(undef, nobs)
#     for i in 1:nobs
#         amb = X[i,:] - prob_means
#         z[i] = multipl * exp(-0.5 * transpose(amb) * AplusBinv * amb)[1,1]
#     end
#     expectation = transpose(z) * invK * f
#     var = cov_func_parameters.w_0 * det(2 * invA * covar +diagm(0 => ones(ndim)))^(-0.5) - transpose(z) * invK * z
#     return (expectation = expectation, variance = var)
# end

"""
    integrate(modl::KrigingModel, prob_means::AbstractArray{U,1}, covar::Hermitian) where U<:Real
Returns the expectation and variance of the integral of a kriging model given the probabilities described my a multivariate normal with means (in each dimension) of prob_means and covariance matrix covar
The integration performed is: int_{x in X} f(x) p(x) dx
Where f(x) is the function which is approximated in the kriging map by an exponential covariance function and p(x) is
the pdf which is multivariate gaussian.
"""
function integrate(modl::KrigingModel, prob_means::AbstractArray{U,1}, covar::Hermitian) where U<:Real
    ndim = length(modl.hyper.w_i)
    nobs = size(modl.X)[1]
    A = diagm(0 => modl.hyper.w_i.^2)
    invA = inv(A)
    AplusBinv = inv(A + covar)

    multipl = modl.hyper.w_0 * LinearAlgebra.det(invA * covar + diagm(0 => ones(ndim)))^(-0.5)
    z = Array{promote_type(U,typeof(AplusBinv),typeof(multipl)),1}(undef, nobs)
    for i in 1:nobs
        amb = modl.X[i,:] - prob_means
        z[i] = multipl * exp(-0.5 * transpose(amb) * AplusBinv * amb)[1,1]
    end
    expectation = transpose(z) * modl.invK * modl.f
    var = modl.hyper.w_0 * det(2 * invA * covar +diagm(0 => ones(ndim)))^(-0.5) - transpose(z) * modl.invK * z
    return (expectation = expectation, variance = var)
end

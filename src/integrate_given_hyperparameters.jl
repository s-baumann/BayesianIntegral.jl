"""
    integrate(modl::KrigingModel, prob_means::AbstractArray{U,1}, covar::Hermitian) where U<:Real
Returns the expectation and variance of the integral of a kriging model given the probabilities described by a multivariate normal with means (in each dimension) of prob_means and covariance matrix covar.
The integration performed is: int_{x in X} f(x) p(x) dx
Where f(x) is the function which is approximated in the kriging map by an exponential covariance function and p(x) is
the pdf which is multivariate gaussian.
"""
function integrate(modl::KrigingModel, prob_means::AbstractArray{U,1}, covar::Hermitian) where U<:Real
    ndim = length(modl.hyper.w_i)
    nobs = size(modl.X)[1]
    A = Diagonal(modl.hyper.w_i .^ 2)
    invA = inv(A)
    AplusBinv = inv(A + covar)

    multipl = modl.hyper.w_0 * det(invA * covar + I)^(-0.5)
    z = Array{promote_type(U,typeof(AplusBinv),typeof(multipl)),1}(undef, nobs)
    amb = similar(prob_means, ndim)
    tmp = similar(amb)
    for i in 1:nobs
        amb .= @view(modl.X[i,:]) .- prob_means
        mul!(tmp, AplusBinv, amb)
        z[i] = multipl * exp(-0.5 * dot(amb, tmp))
    end
    expectation = modl.mu + dot(z, modl.invK_y_m_mu)
    var = modl.sigma2 * (modl.hyper.w_0 * det(2 * invA * covar + I)^(-0.5) - dot(z, modl.invK * z))
    return (expectation = expectation, variance = var)
end

"""
    GaussianKernelHyperparameters
This contains the hyperparameters for the gaussian kernel. The functional form for correlation between x_p and x_q   is   w_0 exp( -0.5 * \\sum_{d=1}^D ((x_{p,d} - x_{q,d})/w_d)^2)
    where D is the number of dimensions. and each term of the summation is a different dimension of x_p and x_q.
"""
struct GaussianKernelHyperparameters{T<:Real}
    w_0::T
    w_i::Array{T,1}
    function GaussianKernelHyperparameters(w_0::S,w_i::Array{G,1}) where S<:Real where G<:Real
        promo = promote_type(S,G)
        return new{promo}(promo(w_0), promo.(w_i))
    end
    function GaussianKernelHyperparameters{S}(aa::GaussianKernelHyperparameters{G}) where S<:Real where G<:Real
        return new{S}(S(aa.w_0), S.(aa.w_i))
    end
end

"""
    KrigingModel
    Our kriging model is defined by a level \\mu, a set of X coordinates and corresponding y values, a set of hyperparameters and a noise level.
In addition we save a number of matrices in order to faciliate computations. One is
This contains the hyperparameters for the gaussian kernel.
"""

struct KrigingModel{T<:Real,F}
    X::AbstractArray{T,2}
    f::AbstractArray{T,1}
    hyper::GaussianKernelHyperparameters{T}
    cov_func::F
    noise::T
    # Now we have stuff that is calculated from the above. Saved in the model purely for efficiency reasons.
    K::Hermitian{T}
    invK::Hermitian{T}
    mu::T
    sigma2::T
    y_m_mu::Array{T,1}
    invK_y_m_mu::Array{T,1}
    sum_invK::T
    function KrigingModel(X::AbstractArray{T,2}, f::AbstractArray{R,1}, hyper::GaussianKernelHyperparameters{S};
                          cov_func::F = gaussian_kernel, noise::U = 1000*eps(),
                          K::Hermitian{W} = K_matrix(X, cov_func, hyper, noise), invK::Hermitian{Z} = LinearAlgebra.inv(K)) where {T<:Real, R<:Real, S<:Real, U<:Real, W<:Real, Z<:Real, F}
            tt = promote_type(T, R, S, U, W, Z)

            dim = size(invK)[1]
            mu = tt(sum(invK * f) / sum(invK))
            y_m_mu = f .- mu
            invK_y_m_mu = invK * y_m_mu
            sigma2 = tt(dot(y_m_mu, invK_y_m_mu) / dim)
            sum_invK = tt(sum(invK))
            return new{tt,F}(Array{tt,2}(X), Array{tt,1}(f), GaussianKernelHyperparameters{tt}(hyper) , cov_func, tt(noise), LinearAlgebra.Hermitian(Array{tt,2}(K)), LinearAlgebra.Hermitian(Array{tt,2}(invK)), mu, sigma2, y_m_mu, Array{tt,1}(invK_y_m_mu), sum_invK)
    end
end

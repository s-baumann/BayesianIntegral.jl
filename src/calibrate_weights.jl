"""
    RProp_params
"""
struct RProp_params{T<:Real}
    eta_plus::T
    eta_minus::T
    Delta_min::T
    Delta_max::T
    Delta_0::T
    function RProp_params()
        return new{Float64}(1.2,0.5,1000.0,0.001,1.0)
    end
end

"""
    train_with_RProp(X::Array{Float64,2}, y::Array{Float64,1}, cov_func_parameters::GaussianKernelHyperparameters, MaxIter::Int, noise::Float64, params::RProp_params)
Trains kriging hyperparameters with RProp.
"""
function calibrate_by_ML_with_Rprop(X::AbstractArray{<:Real,2}, y::AbstractArray{R,1}, cov_func_parameters::GaussianKernelHyperparameters, MaxIter::Integer, noise::Real, params::RProp_params) where R<:Real
    # An implementation of this paper https://pdfs.semanticscholar.org/aa65/042ae494455a14811927eb0574871d276454.pdf
    iter = 0
    w_0 = cov_func_parameters.w_0
    w_i = cov_func_parameters.w_i
    threshold = params.Delta_min * params.eta_plus + 10*eps()
    old_0_sign = 1
    old_i_signs = ones(length(w_i))
    while iter < MaxIter
        marginal_likelihood, K, invK = marginal_likelihood_gaussian_derivatives(X, y, GaussianKernelHyperparameters(w_0, w_i), noise)
        Delta0 = params.Delta_0
        Deltas = Array{typeof(w_0),1}(undef, length(w_i)) .= params.Delta_0
        new_0_sign = sign(marginal_likelihood[1])
        new_i_signs = sign.(marginal_likelihood[2:length(marginal_likelihood)])
        w_0 = w_0 + new_0_sign*Delta0
        w_i = w_i .+ new_i_signs .* Deltas
        if iter > 0
            Delta0 = new_0_sign * old_0_sign > 0 ? Delta0*params.eta_plus : Delta0*params.eta_minus
            Delta0 = min(max(Delta0, params.Delta_min ), params.Delta_max )
            for i in 1:length(w_i)
                Deltas[i] = new_i_signs[i]*old_i_signs[i] > 0 ? Deltas[i]*params.eta_plus : Deltas[i] *params.eta_minus
                Deltas[i] = min(max(Deltas[i], params.Delta_min ), params.Delta_max )
            end
        end
        if all(Deltas .< threshold)
            return GaussianKernelHyperparameters(w_0, w_i)
        end
        old_0_sign  = new_0_sign
        old_i_signs = new_i_signs
        iter = iter + 1
    end
    return GaussianKernelHyperparameters(w_0, w_i)
end

"""
    sample(twister::MersenneTwister, dim::Integer, batch_size::Integer)
This does sampling with or without replacement.
"""
function sample(twister::MersenneTwister, dim::Integer, batch_size::Integer)
    return randperm(twister, dim)[1:batch_size]
end

"""
    calibrate_by_ML_with_SGD(X::Array{Float64,2}, y::Array{Float64,1}, cov_func_parameters::GaussianKernelHyperparameters, steps::Int, batch_size::Int, step_multiple::Float64, noise::Float64, seed::Int = 1988)
This trains a kriging model by using maximum likelihood with stochastic gradient descent.
"""
function calibrate_by_ML_with_SGD(X::AbstractArray{<:Real,2}, y::AbstractArray{R,1}, cov_func_parameters::GaussianKernelHyperparameters, steps::Integer, batch_size::Integer,
                                  step_multiple::Real = 0.05, noise::Real = 0.00001, twister::MersenneTwister = MersenneTwister(1988)) where R<:Real
    ow_0 = cov_func_parameters.w_0
    ow_i = cov_func_parameters.w_i
    ndims = length(ow_i)
    nobs = length(y)
    for s in 1:steps
        samples = sample(twister, nobs, batch_size)
        XSample = X[samples, :]
        ySample = y[samples]
        marginal_likelihood, K, invK = marginal_likelihood_gaussian_derivatives(XSample, ySample, GaussianKernelHyperparameters(ow_0, ow_i), noise)
        normalised_grad =  sign.(marginal_likelihood) .* ( abs.(marginal_likelihood) ./ (abs.(marginal_likelihood) .+ maximum(abs.(marginal_likelihood))) )
        ow_0 = ow_0 .* (1 .+ normalised_grad[1] .* step_multiple)
        ow_i = ow_i .* (1 .+ normalised_grad[2:(ndims+1)] .* step_multiple)
    end
    return GaussianKernelHyperparameters(ow_0, ow_i)
end

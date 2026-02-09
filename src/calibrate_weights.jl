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
        return new{Float64}(1.2,0.5,0.001,1000.0,1.0)
    end
end

"""
    calibrate_by_ML_with_Rprop(X, y, cov_func_parameters, MaxIter, noise, params)
Trains kriging hyperparameters by maximising marginal likelihood with RProp.
"""
function calibrate_by_ML_with_Rprop(X::AbstractArray{<:Real,2}, y::AbstractArray{R,1}, cov_func_parameters::GaussianKernelHyperparameters, MaxIter::Integer, noise::Real, params::RProp_params) where R<:Real
    # An implementation of this paper https://pdfs.semanticscholar.org/aa65/042ae494455a14811927eb0574871d276454.pdf
    iter = 0
    w_0 = cov_func_parameters.w_0
    w_i = copy(cov_func_parameters.w_i)
    threshold = params.Delta_min * params.eta_plus + 10*eps()
    old_0_sign = 1
    old_i_signs = ones(length(w_i))
    new_i_signs = similar(old_i_signs)
    Delta0 = params.Delta_0
    Deltas = fill(params.Delta_0, length(w_i))
    while iter < MaxIter
        marginal_likelihood, K, invK = marginal_likelihood_gaussian_derivatives(X, y, GaussianKernelHyperparameters(w_0, w_i), noise)
        new_0_sign = sign(marginal_likelihood[1])
        new_i_signs .= sign.(@view(marginal_likelihood[2:end]))
        if iter > 0
            Delta0 = new_0_sign * old_0_sign > 0 ? Delta0*params.eta_plus : Delta0*params.eta_minus
            Delta0 = min(max(Delta0, params.Delta_min ), params.Delta_max )
            for i in 1:length(w_i)
                Deltas[i] = new_i_signs[i]*old_i_signs[i] > 0 ? Deltas[i]*params.eta_plus : Deltas[i] *params.eta_minus
                Deltas[i] = min(max(Deltas[i], params.Delta_min ), params.Delta_max )
            end
        end
        w_0 = w_0 + new_0_sign*Delta0
        w_i .+= new_i_signs .* Deltas
        if all(Deltas .< threshold) && Delta0 < threshold
            return GaussianKernelHyperparameters(w_0, w_i)
        end
        old_0_sign  = new_0_sign
        old_i_signs .= new_i_signs
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
    calibrate_by_ML_with_SGD(X, y, cov_func_parameters, steps, batch_size, step_multiple, noise, twister)
Trains kriging hyperparameters by maximising marginal likelihood with stochastic gradient descent.
"""
function calibrate_by_ML_with_SGD(X::AbstractArray{<:Real,2}, y::AbstractArray{R,1}, cov_func_parameters::GaussianKernelHyperparameters, steps::Integer, batch_size::Integer,
                                  step_multiple::Real = 0.05, noise::Real = 0.00001, twister::MersenneTwister = MersenneTwister(1988)) where R<:Real
    ow_0 = cov_func_parameters.w_0
    ow_i = copy(cov_func_parameters.w_i)
    ndims = length(ow_i)
    nobs = length(y)
    for s in 1:steps
        samples = sample(twister, nobs, batch_size)
        XSample = @view X[samples, :]
        ySample = y[samples]
        marginal_likelihood, K, invK = marginal_likelihood_gaussian_derivatives(XSample, ySample, GaussianKernelHyperparameters(ow_0, ow_i), noise)
        abs_ml = abs.(marginal_likelihood)
        normalised_grad = marginal_likelihood ./ (abs_ml .+ maximum(abs_ml))
        ow_0 *= 1 + normalised_grad[1] * step_multiple
        for i in 1:ndims
            ow_i[i] *= 1 + normalised_grad[i+1] * step_multiple
        end
    end
    return GaussianKernelHyperparameters(ow_0, ow_i)
end

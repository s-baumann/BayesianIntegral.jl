"""
    evaluate(modl::KrigingModel, PointToExamine::AbstractArray{T,1}) where T<:Real
Returns the kriging predictor value (ordinary kriging estimate) at the given point.
"""
function evaluate(modl::KrigingModel, PointToExamine::AbstractArray{T,1}; k = correlation_vector_of_a_point(PointToExamine,  modl.X,  modl.cov_func, modl.hyper)) where T<:Real
    yhat =  modl.mu + dot(k, modl.invK_y_m_mu)
    return yhat
end

"""
    predicted_error(modl::KrigingModel, PointToExamine::AbstractArray{T,1}) where T<:Real
Returns the predicted standard error (square root of kriging variance) at the given point.
Uses Equation 9 of Jones, Schonlau, Welch with ordinary kriging correction.
"""
function predicted_error(modl::KrigingModel, PointToExamine::AbstractArray{T,1}; k = correlation_vector_of_a_point(PointToExamine,  modl.X,  modl.cov_func, modl.hyper)) where T<:Real
    # Equation 9 of Jones Schonlau, Welch
    Kinvk = modl.invK * k
    bracketed_term = max(eps(), modl.hyper.w_0 + ((1 - sum(Kinvk))^2) / modl.sum_invK - dot(k, Kinvk))
    return sqrt(modl.sigma2 * bracketed_term)
end

function (modl::KrigingModel)(PointToExamine::AbstractArray{T,1}) where T<:Real
    k = correlation_vector_of_a_point(PointToExamine,  modl.X,  modl.cov_func, modl.hyper)
    value = evaluate(modl, PointToExamine; k = k)
    error = predicted_error(modl, PointToExamine; k = k)
    return (value = value, standard_error = error)
end

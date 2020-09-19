function evaluate(modl::KrigingModel, PointToExamine::Array{T,1}; k = correlation_vector_of_a_point(PointToExamine,  modl.X,  modl.cov_func, modl.hyper)) where T<:Real
    yhat =  modl.mu + (k') *  modl.invK * modl.y_m_mu
    return yhat
end

function predicted_error(modl::KrigingModel, PointToExamine::Array{T,1}; k = correlation_vector_of_a_point(PointToExamine,  modl.X,  modl.cov_func, modl.hyper)) where T<:Real
    # Equation 9 of Jones Schonlau, Welch
    dim = size(modl.invK)[1]
    Kinvk = modl.invK * k
    bracketed_term = max(eps(), 1 + ((1 - sum(Kinvk))^2) / sum(modl.invK * ones(dim)) - ((k') * Kinvk))
    return sqrt(modl.sigma2 * bracketed_term)
end

function (modl::KrigingModel)(PointToExamine::Array{T,1}) where T<:Real
    k = correlation_vector_of_a_point(PointToExamine,  modl.X,  modl.cov_func, modl.hyper)
    value = evaluate(modl, PointToExamine; k = k)
    error = predicted_error(modl, PointToExamine; k = k)
    return (value = value, standard_error = error)
end

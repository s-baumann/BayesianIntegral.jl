function expected_improvement(modl::KrigingModel, fmin::Real, PointToExamine::Array{<:Real,1})
    # This is equation 15 of Jones, Scholau, Welch
    output = modl(PointToExamine)
    fmin_m_y = fmin - output.value
    fmin_m_y_on_s = fmin_m_y / output.standard_error
    return fmin_m_y * Distributions.cdf(Normal(), fmin_m_y_on_s) + output.standard_error *  Distributions.pdf(Normal(), fmin_m_y_on_s)
end

function get_next_query_point_through_expected_improvement(modl::KrigingModel, lower =  repeat([0.0], size(modl.X)[2]), upper = repeat([1.0], size(modl.X)[2]) )
    fmin = minimum(modl.f)
    opt = optimize(x -> -expected_improvement(modl, fmin, x), lower, upper, (lower .+ upper) ./ 2, SAMIN(), Optim.Options(iterations = 10000))
    return opt.minimizer
end

function get_predicted_minimum(modl::KrigingModel, lower =  repeat([0.0], size(modl.X)[2]), upper = repeat([1.0], size(modl.X)[2]) )
    opt = optimize(x -> -modl(x).value, lower, upper, (lower .+ upper) ./ 2, SAMIN(), Optim.Options(iterations = 10000))
    return opt.minimizer
end

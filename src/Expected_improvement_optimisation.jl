"""
    expected_improvement(modl::KrigingModel, fmin::Real, PointToExamine::AbstractArray{<:Real,1})
Computes the expected improvement (Jones, Schonlau, Welch Equation 15) at a point.
Returns how much improvement over the current best value `fmin` is expected.
"""
function expected_improvement(modl::KrigingModel, fmin::Real, PointToExamine::AbstractArray{<:Real,1})
    # This is equation 15 of Jones, Scholau, Welch
    output = modl(PointToExamine)
    fmin_m_y = fmin - output.value
    fmin_m_y_on_s = fmin_m_y / output.standard_error
    return fmin_m_y * Distributions.cdf(Normal(), fmin_m_y_on_s) + output.standard_error *  Distributions.pdf(Normal(), fmin_m_y_on_s)
end

"""
    get_next_query_point_through_expected_improvement(modl::KrigingModel, lower, upper)
Finds the point that maximises expected improvement over the current best observed value.
Uses SAMIN optimisation over the box `[lower, upper]`.
"""
function get_next_query_point_through_expected_improvement(modl::KrigingModel, lower = zeros(size(modl.X)[2]), upper = ones(size(modl.X)[2]) )
    fmin = minimum(modl.f)
    opt = optimize(x -> -expected_improvement(modl, fmin, x), lower, upper, (lower .+ upper) ./ 2, SAMIN(), Optim.Options(iterations = 10000))
    return opt.minimizer
end

"""
    get_predicted_minimum(modl::KrigingModel, lower, upper)
Finds the predicted minimum of the kriging model over the box `[lower, upper]`.
Uses SAMIN optimisation.
"""
function get_predicted_minimum(modl::KrigingModel, lower = zeros(size(modl.X)[2]), upper = ones(size(modl.X)[2]) )
    opt = optimize(x -> modl(x).value, lower, upper, (lower .+ upper) ./ 2, SAMIN(), Optim.Options(iterations = 10000))
    return opt.minimizer
end

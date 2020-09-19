using Random
using BayesianIntegral
using Distributions
using LinearAlgebra
using Statistics

twister = MersenneTwister(10)
X = rand(twister, 20,2)
y = X[:,1] .- (X[:,2] .^ 2) .+ 12
noise = 0.04

hyper = GaussianKernelHyperparameters(1.0, [0.1,0.1])
modl = KrigingModel(X, y, hyper)

# Testing it.
output = modl(X[1,:])
output.standard_error < 1e-6
output = modl(X[2,:])
output.standard_error < 1e-6
output = modl([0.5,0.5])
output.standard_error > 0.001
output = modl([0.75,0.25])
output.standard_error > 0.001

################################################################################

PointToExamine = [0.44, 0.44]
fmin = minimum(modl.f)

EI = expected_improvement(modl, fmin, PointToExamine)
EI2 = expected_improvement(modl, fmin + 0.1, PointToExamine)
EI < EI2


xx = get_next_query_point_through_expected_improvement(modl)

hyper = GaussianKernelHyperparameters(1.0, [0.1,0.1])
batch_size = Integer(floor(size(X)[1]/2))
step_multiple = 0.05
noise = 0.0001
steps = 1000
calibrated_parameters = calibrate_by_ML_with_SGD(X, y, hyper, steps, batch_size, step_multiple, noise)
modl = KrigingModel(X, y, calibrated_parameters; noise  = noise)
xx = get_next_query_point_through_expected_improvement(modl)

# Testing integral
dist = MultivariateNormal([0.5,0.5], Symmetric([1.0 0.0; 0.0 1.0]))
int = integrate(modl, [0.5,0.5], Symmetric([1.0 0.0; 0.0 1.0])).expectation
samples = 100000
numerical_int = zeros(samples)
for i in 1:samples
    x_ = rand(twister, dist, 1)
    numerical_int[i] =  modl(x_[:,1]).value
end
numerical_int = Statistics.mean(numerical_int)

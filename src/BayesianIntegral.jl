module BayesianIntegral

using Distributions
using LinearAlgebra
using Random
using Optim

# Basic operation of Kriging model
include("kriging_structs.jl")
export KrigingModel, GaussianKernelHyperparameters
include("making_covariances.jl")
export gaussian_kernel, marginal_gaussian_kernel, K_matrix
export marginal_likelihood_gaussian_derivatives, log_likelihood
export correlation_vector_of_a_point
include("evaluate_kriging_model.jl")
export evaluate, predicted_error
# Integration of Kriging Model
include("integrate_given_hyperparameters.jl")
export integrate
# Calibration of Kriging Model
include("calibrate_weights.jl")
export calibrate_by_ML_with_SGD, RProp_params, calibrate_by_ML_with_Rprop

# Searching for Optima.
include("Expected_improvement_optimisation.jl")
export expected_improvement, get_next_query_point_through_expected_improvement, get_predicted_minimum

end

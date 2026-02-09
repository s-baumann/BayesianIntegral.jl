using BayesianIntegral
using LinearAlgebra
using Random
using Test

# Deterministic setup shared by all regression tests
rng = MersenneTwister(42)
X_reg = rand(rng, 10, 2)
y_reg = X_reg[:,1] .- (X_reg[:,2] .^ 2) .+ 5.0
hyper_reg = GaussianKernelHyperparameters(1.5, [0.3, 0.4])
noise_reg = 0.01

@testset "Regression Tests" begin

    @testset "gaussian_kernel" begin
        x1 = X_reg[1,:]
        x2 = X_reg[2,:]
        gk = gaussian_kernel(x1, x2, hyper_reg)
        @test gk ≈ 0.045619693403582434 atol=1e-12
    end

    @testset "marginal_gaussian_kernel" begin
        x1 = X_reg[1,:]
        x2 = X_reg[2,:]
        mgk = marginal_gaussian_kernel(x1, x2, hyper_reg)
        @test mgk.cov ≈ 0.045619693403582434 atol=1e-12
        @test mgk.marginal_covariances[1] ≈ 0.030413128935721623 atol=1e-12
        @test mgk.marginal_covariances[2] ≈ 0.7058441282489638 atol=1e-12
        @test mgk.marginal_covariances[3] ≈ 0.2673376805070645 atol=1e-12
    end

    @testset "K_matrix" begin
        K = K_matrix(X_reg, gaussian_kernel, hyper_reg, noise_reg)
        @test size(K) == (10, 10)
        @test K[1,1] ≈ 1.51 atol=1e-14      # w_0 + noise = 1.5 + 0.01
        @test K[1,2] ≈ 0.045619693403582434 atol=1e-12
        @test K[2,1] ≈ 0.045619693403582434 atol=1e-12  # symmetry
        @test K[5,5] ≈ 1.51 atol=1e-14
        @test K[1,10] ≈ 0.6983819183950963 atol=1e-12
    end

    @testset "correlation_vector_of_a_point" begin
        test_point = [0.5, 0.5]
        cv = correlation_vector_of_a_point(test_point, X_reg, gaussian_kernel, hyper_reg)
        @test length(cv) == 10
        @test cv[1] ≈ 0.8749206869708288 atol=1e-12
        @test cv[2] ≈ 0.38979735435707363 atol=1e-12
        @test cv[3] ≈ 1.4258004036464147 atol=1e-12
        @test cv[7] ≈ 1.3081948829918142 atol=1e-12
        @test cv[10] ≈ 1.4707397888092768 atol=1e-12
    end

    @testset "log_likelihood" begin
        K = K_matrix(X_reg, gaussian_kernel, hyper_reg, noise_reg)
        ll = log_likelihood(y_reg, K)
        @test ll ≈ -17.038639280411736 atol=1e-10
    end

    @testset "marginal_likelihood_gaussian_derivatives" begin
        mld = marginal_likelihood_gaussian_derivatives(X_reg, y_reg, hyper_reg, noise_reg)
        @test mld.marginal_likelihoods[1] ≈ 12.959835645479078 atol=1e-8
        @test mld.marginal_likelihoods[2] ≈ 41.20993241215721 atol=1e-8
        @test mld.marginal_likelihoods[3] ≈ 29.798721700895143 atol=1e-8
        @test mld.k_mat[1,1] ≈ 1.51 atol=1e-14
        @test mld.k_mat[1,2] ≈ 0.045619693403582434 atol=1e-12
        @test mld.inv_K_matrix[1,1] ≈ 2.0079992037471817 atol=1e-10
        @test mld.inv_K_matrix[1,2] ≈ 0.49532743137937263 atol=1e-10
    end

    @testset "KrigingModel construction" begin
        modl = KrigingModel(X_reg, y_reg, hyper_reg; noise = noise_reg)
        @test modl.mu ≈ 4.965745187607353 atol=1e-10
        @test modl.sigma2 ≈ 0.08540276850922607 atol=1e-10
        @test modl.K[1,1] ≈ 1.51 atol=1e-14
        @test modl.K[1,2] ≈ 0.045619693403582434 atol=1e-12
        @test modl.invK[1,1] ≈ 2.0079992037471817 atol=1e-10
        @test modl.invK[1,2] ≈ 0.49532743137937263 atol=1e-10
    end

    modl_reg = KrigingModel(X_reg, y_reg, hyper_reg; noise = noise_reg)

    @testset "evaluate" begin
        ev = evaluate(modl_reg, [0.5, 0.5])
        @test ev ≈ 5.250857737518842 atol=1e-10

        ev_00 = evaluate(modl_reg, [0.0, 0.0])
        @test ev_00 ≈ 5.009482923698842 atol=1e-10
    end

    @testset "predicted_error" begin
        pe = predicted_error(modl_reg, [0.5, 0.5])
        @test pe ≈ 0.030242733055479778 atol=1e-10

        pe_00 = predicted_error(modl_reg, [0.0, 0.0])
        @test pe_00 ≈ 0.08787727552802008 atol=1e-10
    end

    @testset "callable interface modl(point)" begin
        output = modl_reg([0.5, 0.5])
        @test output.value ≈ 5.250857737518842 atol=1e-10
        @test output.standard_error ≈ 0.030242733055479778 atol=1e-10

        output_00 = modl_reg([0.0, 0.0])
        @test output_00.value ≈ 5.009482923698842 atol=1e-10
        @test output_00.standard_error ≈ 0.08787727552802008 atol=1e-10
    end

    @testset "integrate" begin
        prob_means = [0.5, 0.5]
        covar = Hermitian([1.0 0.0; 0.0 1.0])
        integ = integrate(modl_reg, prob_means, covar)
        @test integ.expectation ≈ 4.97604257679532 atol=1e-10
        @test integ.variance ≈ 0.003936355385340289 atol=1e-10
    end

    @testset "calibrate_by_ML_with_SGD" begin
        sgd_twister = MersenneTwister(1988)
        init_hyper = GaussianKernelHyperparameters(1.0, [0.5, 0.5])
        sgd_result = calibrate_by_ML_with_SGD(X_reg, y_reg, init_hyper, 100, 5, 0.05, noise_reg, sgd_twister)
        @test sgd_result.w_0 ≈ 9.998675983781643 atol=1e-10
        @test sgd_result.w_i[1] ≈ 2.9356237648994963 atol=1e-10
        @test sgd_result.w_i[2] ≈ 2.5941741666496774 atol=1e-10
    end

    @testset "calibrate_by_ML_with_Rprop" begin
        rprop_params = RProp_params()
        rprop_init = GaussianKernelHyperparameters(1.0, [0.5, 0.5])
        rprop_result = calibrate_by_ML_with_Rprop(X_reg, y_reg, rprop_init, 100, noise_reg, rprop_params)
        @test rprop_result.w_0 ≈ 17.6838961045025 atol=1e-10
        @test rprop_result.w_i[1] ≈ 3.9860245973255934 atol=1e-10
        @test rprop_result.w_i[2] ≈ 3.488367811128867 atol=1e-10
    end

    @testset "expected_improvement" begin
        fmin = minimum(modl_reg.f)
        @test fmin ≈ 4.413803539723226 atol=1e-10

        # At a point near training data: EI should be essentially zero
        ei_near = expected_improvement(modl_reg, fmin, [0.5, 0.5])
        @test ei_near ≈ 0.0 atol=1e-10

        # At a point far from training data
        ei_far = expected_improvement(modl_reg, fmin, [0.0, 0.0])
        @test ei_far ≈ 7.558846092863149e-14 atol=1e-20
    end

    @testset "get_predicted_minimum" begin
        pred_min = get_predicted_minimum(modl_reg)
        # SAMIN is stochastic — use loose tolerance
        # True minimum of x1 - x2^2 + 5 on [0,1]^2 is near (0, 1)
        @test pred_min[1] ≈ 0.04 atol=0.05
        @test pred_min[2] ≈ 1.0 atol=0.01
    end

    @testset "get_next_query_point_through_expected_improvement" begin
        next_pt = get_next_query_point_through_expected_improvement(modl_reg)
        # SAMIN is stochastic — use loose tolerance
        @test next_pt[1] ≈ 0.035 atol=0.05
        @test next_pt[2] ≈ 1.0 atol=0.01
    end

end

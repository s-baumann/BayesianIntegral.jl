using StuTrade
using Test

# Run tests
println("Test Bayes")
@time @test include("test_Bayes.jl")
println("Testing Cubes")
@time @test include("test_cubes.jl")
println("Test marginal gaussian likelihoods")
@time @test include("test_marginal_gaussian_likelihood.jl")

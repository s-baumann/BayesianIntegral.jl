using StuTrade
using Test

# Run tests
println("Test Bayes")
@time @test include("test_Bayes.jl")

using MultivariatePolynomials
using LinearAlgebra
using Distributions
using Random
using Sobol
using BayesianIntegral
Random.seed!(1)

function generate_random_PSD_matrix(n, randGen, sumTraceTarget = 5)
   A = rand(randGen,n,n)
   A = (A*transpose(A))
   sumTrace = sum(diag(A))
   A = A * (sumTraceTarget/sumTrace)
   return A
end

function random_testCase(dims ; termsGen = Poisson(3) , powersGen = Normal(2,1), meansGen = Normal(2,1), random_matrixGen = Exponential(2))
    terms = rand(termsGen) + 1
    units = Array{MultivariatePolynomialUnit}(undef, terms)
    dimNames = string.("x", 1:dims)
    for i in 1:terms
        units[i] = MultivariatePolynomialUnit(rand(powersGen), Dict(dimNames .=> rand(powersGen,dims)))
    end
    poly = MultivariatePolynomial(units)
    UpperLims = Dict(dimNames .=> convert(Array{Float64}, 1:dims))
    LowerLims = Dict(dimNames .=> repeat([0.0],dims))
    integ = evaluate_integral(poly, UpperLims, LowerLims)
    # Random pdf
    means             = rand(meansGen, dims)
    covariance_matrix = Symmetric(generate_random_PSD_matrix(dims, random_matrixGen))
    dist = MvNormal(means, covariance_matrix)
    chol = cholesky(covariance_matrix)
    return poly, UpperLims, LowerLims, integ, means, covariance_matrix, dist, chol
end



function ff(X, poly, dist = nothing)
    dims = length(X)
    dimNames    = string.("x", 1:dims)
    coordinates = Dict(dimNames .=> X)
    poly_val    = evaluate(poly, coordinates)
    if dist != nothing
        pdf_val     = pdf(dist, X)
        return poly_val / pdf_val
    else
        return poly_val
    end
end

function make_X(num, tops, bottoms, random = false)
    dims = length(tops)
    if random
        XRaw = rand(num,dims)
    else
        XRaw = Array{Float64}(undef,num,dims)
        s = SobolSeq(dims)
        for i in 1:num
            XRaw[i,:] = next!(s)
        end
    end
    gaps = tops - bottoms
    for c in 1:dims
        XRaw[:,c] = bottoms[c] .+ (gaps[c] .* XRaw[:,c])
    end
    return XRaw
end


#
dims = 2
poly, UpperLims, LowerLims, integ, means, covariance_matrix, dist, chol = random_testCase(dims)
evals = 100
# QMC
X = make_X(evals, convert(Array{Float64}, 1:dims), repeat([0.0],dims))
y = Array{Float64}(undef,evals)
for i in 1:evals
    y[i] = ff(X[i,:], poly)
end
mean(y) - (integ/factorial(dims))# Because the area we are integrating is this size.
integ / mean(y)
# BMC
w_0 = 1.0
w_i = repeat([1.0],dims)
noise = 0.2
y = Array{Float64}(undef,evals)
for i in 1:evals
    y[i] = ff(X[i,:], poly, dist)
end
bayesian_integral_exponential(X , y, means, covariance_matrix, w_0, w_i, noise)

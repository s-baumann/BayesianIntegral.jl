using BayesianIntegral
using Distributions: Normal, MvNormal, pdf
using LinearAlgebra: diagm, Symmetric
using Statistics
using HCubature: hcubature
using Sobol
using Random
standard_normal = Normal(0.0,1.0)

# Integral of gaussian distribution should be 1.
samples = 29
p(x) = 1.0
X = Array{Float64,2}(undef,samples,1)
X[:,1] =  collect(-2.0:(4.0/(samples-1)):2.0) #
y = p.(X)[:,1]
w_0 = 1.0
w_i = [1.0]
prob_means = [0.0]
covar = Array{Float64,2}(undef,1,1)
covar[1,1] = 1.0
covar = Symmetric(covar)
noise = 0.2
integ = bayesian_integral_exponential( X, y , prob_means , covar, w_0, w_i, noise )

# Integral in two dimensions
samples = 25
dims = 2
p(x) = 1.0
s = SobolSeq(dims)
X = convert( Array{Float64}, hcat([next!(s, repeat([0.5] , outer = dims)     ) for i = 1:samples]...)' )
y = repeat([1.0] , outer = samples)
w_0 = 1.0
w_i = repeat([10.0] , outer = dims)
prob_means = repeat([0.0] , outer = dims)
covar = diagm(0 => ones(dims))
noise = 0.2
bayesian_integral_exponential( X, y , prob_means , covar, w_0, w_i , noise )

# In ten dimensinos
samples = 50
dims = 10
p(x) = 1.0
s = SobolSeq(dims)
X = convert(Array{Float64}, hcat([next!(s, repeat([0.5] , outer = dims)     ) for i = 1:samples]...)')
y = repeat([1.0] , outer = samples)
w_0 = 1.0
w_i = repeat([10.0] , outer = dims)
prob_means = repeat([0.0] , outer = dims)
covar = Symmetric(diagm(0 => ones(dims)))
noise = 0.2
bayesian_integral_exponential( X, y , prob_means , covar, w_0, w_i , noise )

##### More complex cases #####
# Now looking at a more complex one.
function f(x::Array{Float64})
    ndims = length(x)
    total = 12.0
    for i in 1:ndims
        total = total - x[i]^2 - 2*x[i]
    end
    return total
end
function fp(x::Array{Float64})
    ndims = length(x)
    if ndims == 1
        return f(x) * pdf(standard_normal, x)
    else
        dist = MvNormal(zeros(ndims),diagm(0 => ones(ndims)))
        return f(x) * pdf(dist, x)
    end
end
function fp(x)
    global counter = counter + 1
    flipped = convert(Array{Float64,1},x)
    return fp(flipped)
end

function compare_for_f(dims, bayesianAttempts = 0, paths = 0;  seed = 1988)
    Random.seed!(1234)
    # Traditional
    global counter = 0
    lims = 3.0
    maxIter = 5000
    numerical_val, numerical_err = hcubature(x -> fp(x), repeat([-lims], outer = dims) , repeat([lims], outer = dims), maxevals = maxIter )
    cont = counter

    # Kriging
    if bayesianAttempts < 1
        samples = cont
    else
        samples = bayesianAttempts
    end
    s = SobolSeq(dims)
    X = (hcat([next!(s, repeat([0.5] , outer = dims)     ) for i = 1:samples]...)' .- 0.5) .* (lims*2.0)
    y = Array{Float64}(undef, samples)
    for r in 1:samples
        y[r] = f(X[r,:])
    end
    noise = 0.05
    w_0 = 1.0
    w_i = repeat([1.0], outer = dims)
    prob_means = repeat([0.0], outer = dims)
    covar = Symmetric(diagm(0 => ones(dims)))
    bayesian_val, bayesian_err = bayesian_integral_exponential( X, y , prob_means , covar, w_0, w_i, noise )

    # Kriging with new weights
    steps = 2000
    batch_size = convert(Int, floor( samples / 4 ) )
    step_multiple = 0.02
    seed = 1988
    like = bayesian_integral_exponential_log_likelihood( y , X, vcat(w_0,w_i), noise )
    nw_0, nw_i = solve_for_weights_gaussian(X, y, w_0, w_i, steps, batch_size, step_multiple, noise, seed)
    nlike = bayesian_integral_exponential_log_likelihood( y , X, vcat(nw_0,nw_i), noise )
    nbayesian_val, nbayesian_err = bayesian_integral_exponential( X, y , prob_means , covar, nw_0, nw_i, noise )


    # Kriging with new weights and RProp
    params = RProp_params(1.01,0.99,0.2,5.0,0.5)
    MaxIter = 2000
    nnw_0, nnw_i = train_with_RProp(X, y, w_0, w_i, MaxIter, noise, params)
    rproplikelihood = bayesian_integral_exponential_log_likelihood( y , X, vcat(nnw_0,nnw_i), noise )
    rpropnbayesian_val, rpropnbayesian_err = bayesian_integral_exponential( X, y , prob_means , covar, nnw_0, nnw_i, noise )

    # MC Integration
    if paths < 1
        paths = cont
    end
    dist = MvNormal(zeros(dims),diagm(0 => ones(dims)))
    Xs = transpose(convert(Matrix{Float64}, rand(dist, paths)))
    ys = Array{Float64}(undef, paths)
    for r in 1:paths
        ys[r] = f(Xs[r,:])
    end
    MC_integral = mean(ys)
    MC_err = std(ys) / sqrt(paths)

    print("For ", dims, " dimensions \n")
    print("    Bayesian    Integral is               ", bayesian_val     , " with error ", bayesian_err     ," and ", samples , " evaluations \n")
    print("    Calibrated Bayesian Integral is       ", nbayesian_val    , " with error ", nbayesian_err    ," and ", samples , " evaluations \n")
    print("    Rprop Calibrated Bayesian Integral is ", rpropnbayesian_val    , " with error ", rpropnbayesian_err    ," and ", samples , " evaluations \n")
    print("    Traditional Integral is               ", numerical_val[1] , " with error ", numerical_err[1] ," and ", cont    , " evaluations \n")
    print("    MC          Integral is               ", MC_integral      , " with error ", MC_err           ," and ", paths   , " evaluations \n")
    print("        Likelihood was ", like   , " and calibrated is ", nlike, " and Rprop calibrated is ", rproplikelihood  ,"\n")
    print("        weights became ", nw_0   , " and  ", nw_i, " with SGD \n")
    print("        weights became ", nnw_0   , " and  ", nnw_i, " with RProp \n")
    return bayesian_val, bayesian_err, samples, numerical_val[1], numerical_err, cont, MC_integral, MC_err, paths
end

compare_for_f(1, 10)

compare_for_f(2, 30)

compare_for_f(10, 200, 20000)

compare_for_f(20, 2000, 2000)

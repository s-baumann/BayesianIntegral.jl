using Documenter, BayesianIntegral

makedocs(
    format = Documenter.HTML(),
    sitename = "BayesianIntegral",
    modules = [BayesianIntegral],
    pages = ["index.md"]
)

deploydocs(
    repo   = "github.com/s-baumann/BayesianIntegral.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing
)

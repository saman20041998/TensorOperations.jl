using Random
using LinearAlgebra

# Until problems with .+ transforming ranges to arrays are settled
⊞(s::Int, r::StepRange{Int,Int}) = (first(r)+s):step(r):(last(r)+s)
⊞(s::Int, r::UnitRange{Int}) = (first(r)+s):(last(r)+s)

using Test
using TensorOperations

TensorOperations.enable_blas()
TensorOperations.enable_cache()
include("methods.jl")
include("tensor.jl")
TensorOperations.disable_cache()
include("methods.jl")
include("tensor.jl")
TensorOperations.disable_blas()
include("methods.jl")
include("tensor.jl")

include("tensoropt.jl")

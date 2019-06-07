using FastTransforms, Compat
using Compat.Test

if VERSION < v"0.7"
    srand(0)
else
    Random.seed!(0)
end

println()
println("Spherical harmonic transform")
println()

include("slowplantests.jl")

include("fastplantests.jl")

include("thinplantests.jl")

include("pointwisetests.jl")

include("synthesisanalysistests.jl")

include("vectorfieldtests.jl")

include("apitests.jl")

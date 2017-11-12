using FastTransforms, Compat
using Compat.Test

srand(0)

println()
println("Spherical harmonic transform")
println()

include("slowplantests.jl")

include("fastplantests.jl")

include("thinplantests.jl")

include("pointwisetests.jl")

include("synthesisanalysistests.jl")

include("apitests.jl")

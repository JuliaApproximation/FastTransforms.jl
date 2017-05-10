using FastTransforms, Base.Test

include("sphericalharmonictestfunctions.jl")

println("Testing slow plan")
include("test_slowplan.jl")
println("Testing fast plan")
include("test_fastplan.jl")
println("Testing thin plan")
include("test_thinplan.jl")

using FastTransforms, Libdl, LinearAlgebra, Test

include("specialfunctionstests.jl")

include("chebyshevtests.jl")

include("quadraturetests.jl")

if Sys.isapple()
    include("libfasttransformstests.jl")
else
    error("FastTransforms is not properly installed. Please run Pkg.build(\"FastTransforms\") ",
          "and restart Julia.")
end

include("nuffttests.jl")

include("fftBigFloattests.jl")

include("paduatests.jl")

include("gaunttests.jl")

include("toeplitztests.jl")

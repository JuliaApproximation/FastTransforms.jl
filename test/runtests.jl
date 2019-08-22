using FastTransforms, Compat
using Compat.Test, Compat.LinearAlgebra, Compat.Random
if VERSION < v"0.7"
    srand(0)
else
    using Compat.Statistics
    Random.seed!(0)
end

include("basictests.jl")

include("chebyshevtests.jl")

include("nuffttests.jl")

include("chebyshevjacobitests.jl")

include("chebyshevlegendretests.jl")

include("gaunttests.jl")

include("fftBigFloattests.jl")

include("paduatests.jl")

include("butterflytests.jl")

if Libdl.find_library(FastTransforms.libfasttransforms) ≡ FastTransforms.libfasttransforms
    include("libfasttransformstests.jl")
end

include("sphericalharmonics/sphericalharmonictests.jl")

include("toeplitztests.jl")

include("specialfunctionstests.jl")

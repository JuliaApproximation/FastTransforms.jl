using FastTransforms
if VERSION < v"0.7-"
    using Base.Test
else
    using Test
end

srand(0)

include("basictests.jl")

include("nuffttests.jl")

include("chebyshevjacobitests.jl")

include("chebyshevlegendretests.jl")

include("gaunttests.jl")

include("fftBigFloattests.jl")

include("paduatests.jl")

include("butterflytests.jl")

include("sphericalharmonics/sphericalharmonictests.jl")

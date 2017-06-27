using FastTransforms, Base.Test

srand(0)

include("basictests.jl")

include("chebyshevjacobitests.jl")

include("chebyshevlegendretests.jl")

include("gaunttests.jl")

include("fftBigFloattests.jl")

include("paduatests.jl")

include("butterflytests.jl")

include("sphericalharmonics/sphericalharmonictests.jl")

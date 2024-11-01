using FastTransforms, LinearAlgebra, Test

import FastTransforms: normest

@testset "ToeplitzPlusHankel" begin
    n = 128
    for T in (Float32, Float64)
        μ = FastTransforms.chebyshevmoments1(T, 2n-1)
        G = ChebyshevGramMatrix(μ)
        TpH = ToeplitzPlusHankel(G)
        @test TpH ≈ G
        @test norm(TpH) ≤ normest(TpH)
        @test normest(TpH) == normest(G)
    end
end

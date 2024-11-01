using FastTransforms, LinearAlgebra, Test

@testset "ToeplitzPlusHankel" begin
    n = 128
    for T in (Float32, Float64)
        μ = FastTransforms.chebyshevmoments1(T, 2n-1)
        G = ChebyshevGramMatrix(μ)
        TpH = ToeplitzPlusHankel(G)
        @test TpH ≈ G
    end
end

using BlockArrays, FastTransforms, LazyArrays, LinearAlgebra, Test

import FastTransforms: chebyshevmoments1, chebyshevabsmoments1, bivariatemoments

@testset "BivariateGramMatrix" begin
    n = 20
    # w(x, y) = 1 + |x| + |y| - |xy|
    μ1 = chebyshevmoments1(Float64, 2n-1)
    μ2 = chebyshevabsmoments1(Float64, 2n-1)
    μ = bivariatemoments(μ1, μ1) + bivariatemoments(μ2, μ1) + bivariatemoments(μ1, μ2) - bivariatemoments(μ2, μ2)

    X = FastTransforms._chebyshev_x(Float64, 2n-1)
    Y = FastTransforms._chebyshev_y(Float64, 2n-1)

    W = BivariateGramMatrix(μ, X, Y)
    @test issymmetric(W)
    @test isposdef(W)
    WC = BivariateChebyshevGramMatrix(μ)
    @test W ≈ WC

    R = cholesky(W).U
    RC = cholesky(WC).U

    @test R ≈ RC
end

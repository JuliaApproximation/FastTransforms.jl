using BlockArrays, BlockBandedMatrices, FastTransforms, LazyArrays, LinearAlgebra, Test

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

    Gx = FastTransforms.compute_skew_generators(Val(1), W)
    GxC = FastTransforms.compute_skew_generators(Val(1), WC)
    @test Gx ≈ GxC

    Gy = FastTransforms.compute_skew_generators(Val(2), W)
    GyC = FastTransforms.compute_skew_generators(Val(2), WC)
    @test Gy ≈ GyC

    J = [zeros(n, n) Matrix{Float64}(I, n, n); Matrix{Float64}(-I, n, n) zeros(n, n)]
    @test W.X'W-W*W.X ≈ Gx*J*Gx'
    @test W.Y'W-W*W.Y ≈ Gy*J*Gy'

    R = cholesky(W).U
    RC = cholesky(WC).U

    @test R ≈ RC

    μ1 = PaddedVector(1 ./ [1,2,3], 2n-1)
    μ2 = PaddedVector(1 ./ [1,2,3,4,5,6], 2n-1)
    μ = bivariatemoments(μ1, μ2)
    μ̂ = bivariatemoments(Vector(μ1), Vector(μ2))
    @test μ ≈ μ̂

    W = BivariateGramMatrix(μ, X, Y)
    WC = BivariateChebyshevGramMatrix(μ)
    @test blockbandwidths(W) == blockbandwidths(WC) == subblockbandwidths(W) == subblockbandwidths(WC) == (7, 7)
    @test W ≈ WC
end

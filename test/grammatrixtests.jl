using FastTransforms, BandedMatrices, LazyArrays, LinearAlgebra, Test

@testset "GramMatrix" begin
    n = 128
    for T in (Float32, Float64, BigFloat)
        R = plan_leg2cheb(T, n; normcheb=true)*I
        X = Tridiagonal([T(n)/(2n-1) for n in 1:n-1], zeros(T, n), [T(n)/(2n+1) for n in 1:n-1]) # Legendre X
        W = Symmetric(R'R)
        G = GramMatrix(W, X)
        F = cholesky(G)
        @test F.L*F.L' ≈ W
        @test F.U ≈ R

        R = plan_leg2cheb(T, n; normcheb=true, normleg=true)*I
        X = SymTridiagonal(zeros(T, n), [sqrt(T(n)^2/(4*n^2-1)) for n in 1:n-1]) # normalized Legendre X
        W = Symmetric(R'R)
        G = GramMatrix(W, X)
        F = cholesky(G)
        @test F.L*F.L' ≈ W
        @test F.U ≈ R

        b = 4
        X = BandedMatrix(SymTridiagonal(zeros(T, n+b), [sqrt(T(n)^2/(4*n^2-1)) for n in 1:n+b-1])) # normalized Legendre X
        W = I+X^2+X^4
        W = Symmetric(W[1:n, 1:n])
        X̃ = BandedMatrix(SymTridiagonal(zeros(T, n), [sqrt(T(n)^2/(4*n^2-1)) for n in 1:n-1])) # normalized Legendre X
        G = GramMatrix(W, X̃)
        @test bandwidths(G) == (b, b)
        F = cholesky(G)
        @test F.L*F.L' ≈ W

        @test G ≈ GramMatrix(W[1:5, 1], X̃)

        X = BandedMatrix(SymTridiagonal(T[2n-1 for n in 1:n+b], T[-n for n in 1:n+b-1])) # Laguerre X, tests nonzero diagonal
        W = I+X^2+X^4
        W = Symmetric(W[1:n, 1:n])
        X = BandedMatrix(SymTridiagonal(T[2n-1 for n in 1:n], T[-n for n in 1:n-1])) # Laguerre X
        G = GramMatrix(W, X)
        @test bandwidths(G) == (b, b)
        F = cholesky(G)
        @test F.L*F.L' ≈ W
    end
    W = reshape([i for i in 1.0:n^2], n, n)
    X = reshape([i for i in 1.0:4n^2], 2n, 2n)
    @test_throws "different sizes" GramMatrix(W, X)
    X = X[1:n, 1:n]
    @test_throws "nonsymmetric" GramMatrix(W, X)
    @test_throws "nontridiagonal" GramMatrix(Symmetric(W), X)
end

@testset "ChebyshevGramMatrix" begin
    n = 128
    for T in (Float32, Float64, BigFloat)
        μ = FastTransforms.chebyshevmoments1(T, 2n-1)
        G = ChebyshevGramMatrix(μ)
        F = cholesky(G)
        @test F.L*F.L' ≈ G
        R = plan_cheb2leg(T, n; normleg=true)*I
        @test F.U ≈ R

        α, β = (T(0.123), T(0.456))
        μ = FastTransforms.chebyshevjacobimoments1(T, 2n-1, α, β)
        G = ChebyshevGramMatrix(μ)
        F = cholesky(G)
        @test F.L*F.L' ≈ G
        R = plan_cheb2jac(T, n, α, β; normjac=true)*I
        @test F.U ≈ R

        μ = FastTransforms.chebyshevlogmoments1(T, 2n-1)
        G = ChebyshevGramMatrix(μ)
        F = cholesky(G)
        @test F.L*F.L' ≈ G

        μ = FastTransforms.chebyshevabsmoments1(T, 2n-1)
        G = ChebyshevGramMatrix(μ)
        F = cholesky(G)
        @test F.L*F.L' ≈ G

        μ = PaddedVector(T(1) ./ [1,2,3,4,5], 2n-1)
        G = ChebyshevGramMatrix(μ)
        @test bandwidths(G) == (4, 4)
        F = cholesky(G)
        @test F.L*F.L' ≈ G
        μd = Vector{T}(μ)
        Gd = ChebyshevGramMatrix(μd)
        Fd = cholesky(Gd)
        @test F.L ≈ Fd.L
    end
end

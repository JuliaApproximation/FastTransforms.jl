using FastTransforms, BandedMatrices, LazyArrays, LinearAlgebra, Test

@testset "GramMatrix" begin
    n = 128
    for T in (Float32, Float64, BigFloat)
        R = plan_leg2cheb(T, n; normcheb=true)*I
        X = Tridiagonal([T(n)/(2n-1) for n in 1:n-1], zeros(T, n), [T(n)/(2n+1) for n in 1:n-1]) # Legendre X
        W = GramMatrix(Symmetric(R'R), X)
        @test issymmetric(W)
        @test isposdef(W)
        F = cholesky(W)
        @test F.L*F.L' ≈ Symmetric(R'R)
        @test F.U ≈ R

        R = plan_leg2cheb(T, n; normcheb=true, normleg=true)*I
        X = SymTridiagonal(zeros(T, n), [sqrt(T(n)^2/(4*n^2-1)) for n in 1:n-1]) # normalized Legendre X
        W = GramMatrix(Symmetric(R'R), X)
        F = cholesky(W)
        @test F.L*F.L' ≈ Symmetric(R'R)
        @test F.U ≈ R

        b = 4
        X = BandedMatrix(SymTridiagonal(zeros(T, n+b), [sqrt(T(n)^2/(4*n^2-1)) for n in 1:n+b-1])) # normalized Legendre X
        M = Symmetric((I+X^2+X^4)[1:n, 1:n])
        X = BandedMatrix(SymTridiagonal(zeros(T, n), [sqrt(T(n)^2/(4*n^2-1)) for n in 1:n-1])) # normalized Legendre X
        W = GramMatrix(M, X)
        @test bandwidths(W) == (b, b)
        F = cholesky(W)
        @test F.L*F.L' ≈ M

        X = BandedMatrix(SymTridiagonal(T[2n-1 for n in 1:n+b], T[-n for n in 1:n+b-1])) # Laguerre X, tests nonzero diagonal
        M = Symmetric((I+X^2+X^4)[1:n, 1:n])
        X = BandedMatrix(SymTridiagonal(T[2n-1 for n in 1:n], T[-n for n in 1:n-1])) # Laguerre X, tests nonzero diagonal
        W = GramMatrix(M, X)
        @test bandwidths(W) == (b, b)
        F = cholesky(W)
        @test F.L*F.L' ≈ M

        for μ in (PaddedVector([T(4)/3;0;-T(4)/15], 2n-1), # w(x) = 1-x^2
                  PaddedVector([T(26)/15;0;-T(4)/105;0;T(16)/315], 2n-1), # w(x) = 1-x^2+x^4
                  T(1) ./ (1:2n-1)) # Related to a log weight
            X = Tridiagonal([T(n)/(2n-1) for n in 1:2n-2], zeros(T, 2n-1), [T(n)/(2n+1) for n in 1:2n-2]) # Legendre X
            W = GramMatrix(μ, X)
            X = Tridiagonal(X[1:n, 1:n])
            G = FastTransforms.compute_skew_generators(W)
            J = T[0 1; -1 0]
            @test X'W-W*X ≈ G*J*G'
        end
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
        W = ChebyshevGramMatrix(μ)
        F = cholesky(W)
        @test F.L*F.L' ≈ W
        R = plan_cheb2leg(T, n; normleg=true)*I
        @test F.U ≈ R

        α, β = (T(0.123), T(0.456))
        μ = FastTransforms.chebyshevjacobimoments1(T, 2n-1, α, β)
        W = ChebyshevGramMatrix(μ)
        F = cholesky(W)
        @test F.L*F.L' ≈ W
        R = plan_cheb2jac(T, n, α, β; normjac=true)*I
        @test F.U ≈ R

        μ = FastTransforms.chebyshevlogmoments1(T, 2n-1)
        W = ChebyshevGramMatrix(μ)
        F = cholesky(W)
        @test F.L*F.L' ≈ W

        μ = FastTransforms.chebyshevabsmoments1(T, 2n-1)
        W = ChebyshevGramMatrix(μ)
        F = cholesky(W)
        @test F.L*F.L' ≈ W

        μ = PaddedVector(T(1) ./ [1,2,3,4,5], 2n-1)
        W = ChebyshevGramMatrix(μ)
        @test bandwidths(W) == (4, 4)
        F = cholesky(W)
        @test F.L*F.L' ≈ W
        μd = Vector{T}(μ)
        Wd = ChebyshevGramMatrix(μd)
        Fd = cholesky(Wd)
        @test F.L ≈ Fd.L

        X = Tridiagonal([T(1); ones(T, n-2)/2], zeros(T, n), ones(T, n-1)/2)
        G = FastTransforms.compute_skew_generators(W)
        J = T[0 1; -1 0]
        @test 2*(X'W-W*X) ≈ G*J*G'
    end
end

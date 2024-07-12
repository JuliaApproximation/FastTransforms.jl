using BandedMatrices, FastTransforms, LinearAlgebra, ToeplitzMatrices, Test

import FastTransforms: SymmetricToeplitzPlusHankel, SymmetricBandedToeplitzPlusHankel

@testset "SymmetricToeplitzPlusHankel" begin
    n = 128
    for T in (Float32, Float64, BigFloat)
        μ = -FastTransforms.chebyshevlogmoments1(T, 2n-1)
        μ[1] += 1
        W = SymmetricToeplitzPlusHankel(μ/2)
        SMW = Symmetric(Matrix(W))
        @test W ≈ SymmetricToeplitz(μ[1:(length(μ)+1)÷2]/2) + Hankel(μ/2)
        L = cholesky(W).L
        R = cholesky(SMW).U
        @test L*L' ≈ W
        @test L' ≈ R
    end
end

@testset "SymmetricBandedToeplitzPlusHankel" begin
    n = 1024
    for T in (Float32, Float64)
        μ = T[1.875; 0.00390625; 0.5; 0.0009765625; 0.0625]
        W = SymmetricBandedToeplitzPlusHankel(μ/2, n)
        SBW = Symmetric(BandedMatrix(W))
        W1 = SymmetricToeplitzPlusHankel([μ/2; zeros(2n-1-length(μ))])
        SMW = Symmetric(Matrix(W))
        U = cholesky(SMW).U
        L = cholesky(W1).L
        UB = cholesky(SBW).U
        R = cholesky(W).U
        @test L*L' ≈ W
        @test UB'UB ≈ W
        @test R'R ≈ W
        @test UB ≈ U
        @test L' ≈ U
        @test R ≈ U
    end
end

@testset "Fast Cholesky" begin
    n = 128
    for T in (Float32, Float64, BigFloat)
        R = plan_leg2cheb(T, n; normcheb=true)*I
        X = Tridiagonal([T(n)/(2n-1) for n in 1:n-1], zeros(T, n), [T(n)/(2n+1) for n in 1:n-1]) # Legendre X
        W = Symmetric(R'R)
        F = FastTransforms.fastcholesky(W, X)
        @test F.L*F.L' ≈ W
        @test F.U ≈ R
    end
end

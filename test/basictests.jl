using FastTransforms, Base.Test, LowRankApprox

import FastTransforms: Cnλ, Λ, lambertw, Cnαβ, Anαβ, fejer1, fejer2, clenshawcurtis

@testset "Special functions" begin
    n = 0:1000_000
    λ = 0.123
    @time FastTransforms.Cnλ(n,λ)
    @time FastTransforms.Cnλ(n,λ)

    x = linspace(0,20,81)
    @test norm((Λ.(x)-Λ.(big.(x)))./Λ.(x),Inf) < 2eps()
    @test norm((lambertw.(x)-lambertw.(big.(x)))./max.(lambertw.(x),1), Inf) < 2eps()

    x = 0:0.5:10_000
    λ₁,λ₂ = 0.125,0.875
    @test norm((Λ.(x,λ₁,λ₂).-Λ.(big.(x),big(λ₁),big(λ₂)))./Λ.(big.(x),big(λ₁),big(λ₂)),Inf) < 4eps()
    λ₁,λ₂ = 1//3,2//3
    @test norm((Λ.(x,Float64(λ₁),Float64(λ₂))-Λ.(big.(x),big(λ₁),big(λ₂)))./Λ.(big.(x),big(λ₁),big(λ₂)),Inf) < 4eps()

    n = 0:1000
    α = 0.125
    β = 0.375

    @time FastTransforms.Cnαβ(n,α,β);
    @test norm(FastTransforms.Cnαβ(n,α,β)./FastTransforms.Cnαβ(n,big(α),big(β))-1,Inf) < 3eps()

    @time FastTransforms.Anαβ(n,α,β);
    @test norm(FastTransforms.Anαβ(n,α,β)./FastTransforms.Anαβ(n,big(α),big(β))-1,Inf) < 4eps()
end

@testset "Fejer and Clenshaw--Curtis quadrature" begin
    N = 20
    f(x) = exp(x)

    x,w = fejer1(N,0.,0.)
    @test norm(dot(f.(x),w)-2sinh(1)) ≤ 4eps()
    x,w = fejer2(N,0.,0.)
    @test norm(dot(f.(x),w)-2sinh(1)) ≤ 4eps()
    x,w = clenshawcurtis(N,0.,0.)
    @test norm(dot(f.(x),w)-2sinh(1)) ≤ 4eps()

    x,w = fejer1(N,0.25,0.35)
    @test norm(dot(f.(x),w)-2.0351088204147243) ≤ 4eps()
    x,w = fejer2(N,0.25,0.35)
    @test norm(dot(f.(x),w)-2.0351088204147243) ≤ 4eps()
    x,w = clenshawcurtis(N,0.25,0.35)
    @test norm(dot(f.(x),w)-2.0351088204147243) ≤ 4eps()
end

@testset "Allocation-free ID matrix-vector products" begin
    for T in (Float64, Complex128)
        r = rand(T)
        A = idfact([r/(i+j-1) for i in 1:200, j = 1:50])
        P = A[:P]
        k, n = size(A)

        x = rand(T, n)
        y = zeros(T, k)

        @test FastTransforms.A_mul_B!(y, A, P, x, 1, 1) == A*x

        x = rand(T, k)
        y = zeros(T, n)

        @test norm(FastTransforms.At_mul_B!(y, A, P, x, 1, 1) - A.'x, Inf) < 10eps()

        fill!(y, zero(T))
        @test norm(FastTransforms.Ac_mul_B!(y, A, P, x, 1, 1) - A'x, Inf) < 10eps()
    end
end

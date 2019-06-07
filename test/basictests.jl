using Compat, FastTransforms, LowRankApprox
using Compat.Test
import FastTransforms: Cnλ, Λ, lambertw, Cnαβ, Anαβ, pochhammer
import FastTransforms: clenshawcurtisnodes, clenshawcurtisweights, fejernodes1, fejerweights1, fejernodes2, fejerweights2
import FastTransforms: chebyshevmoments1, chebyshevmoments2, chebyshevjacobimoments1, chebyshevjacobimoments2, chebyshevlogmoments1, chebyshevlogmoments2
import Compat: range

@testset "Special functions" begin
    n = 0:1000_000
    λ = 0.123
    @time FastTransforms.Cnλ(n,λ)
    @time FastTransforms.Cnλ(n,λ)

    x =range(0,stop=20,length=81)
    @test norm((Λ.(x) .- Λ.(big.(x)))./Λ.(x),Inf) < 2eps()
    @test norm((lambertw.(x) .- lambertw.(big.(x)))./max.(lambertw.(x),1), Inf) < 2eps()

    x = 0:0.5:10_000
    λ₁,λ₂ = 0.125,0.875
    @test norm((Λ.(x,λ₁,λ₂) .- Λ.(big.(x),big(λ₁),big(λ₂)))./Λ.(big.(x),big(λ₁),big(λ₂)),Inf) < 4eps()
    λ₁,λ₂ = 1//3,2//3
    @test norm((Λ.(x,Float64(λ₁),Float64(λ₂)) .- Λ.(big.(x),big(λ₁),big(λ₂))) ./ Λ.(big.(x),big(λ₁),big(λ₂)),Inf) < 4eps()

    n = 0:1000
    α = 0.125
    β = 0.375

    @time FastTransforms.Cnαβ.(n,α,β);
    @test norm(FastTransforms.Cnαβ.(n,α,β) ./ FastTransforms.Cnαβ.(n,big(α),big(β)) .- 1,Inf) < 3eps()

    @time FastTransforms.Anαβ.(n,α,β);
    @test norm(FastTransforms.Anαβ.(n,α,β) ./ FastTransforms.Anαβ.(n,big(α),big(β)) .- 1,Inf) < 4eps()

    @test pochhammer(2,1) == pochhammer(2,1.0) == pochhammer(2.0,1) == 2
    @test pochhammer(1.1,2.2) ≈ gamma(3.3)/gamma(1.1)
    @test pochhammer(-2,1) == pochhammer(-2,1.0) == pochhammer(-2.0,1) == -2
end

@testset "Fejer and Clenshaw--Curtis quadrature" begin
    N = 20
    f(x) = exp(x)

    x = clenshawcurtisnodes(Float64, N)
    μ = chebyshevmoments1(Float64, N)
    w = clenshawcurtisweights(μ)
    @test norm(dot(f.(x),w)-2sinh(1)) ≤ 4eps()

    μ = chebyshevjacobimoments1(Float64, N, 0.25, 0.35)
    w = clenshawcurtisweights(μ)
    @test norm(dot(f.(x),w)-2.0351088204147243) ≤ 4eps()

    μ = chebyshevlogmoments1(Float64, N)
    w = clenshawcurtisweights(μ)
    @test norm(sum(w./(x .- 3)) - π^2/12) ≤ 4eps()

    x = fejernodes1(Float64, N)
    μ = chebyshevmoments1(Float64, N)
    w = fejerweights1(μ)
    @test norm(dot(f.(x),w)-2sinh(1)) ≤ 4eps()

    μ = chebyshevjacobimoments1(Float64, N, 0.25, 0.35)
    w = fejerweights1(μ)
    @test norm(dot(f.(x),w)-2.0351088204147243) ≤ 4eps()

    μ = chebyshevlogmoments1(Float64, N)
    w = fejerweights1(μ)
    @test norm(sum(w./(x .- 3)) - π^2/12) ≤ 4eps()

    x = fejernodes2(Float64, N)
    μ = chebyshevmoments2(Float64, N)
    w = fejerweights2(μ)
    @test norm(dot(f.(x),w)-2sinh(1)) ≤ 4eps()

    μ = chebyshevjacobimoments2(Float64, N, 0.25, 0.35)
    w = fejerweights2(μ)
    @test norm(dot(f.(x),w)-2.0351088204147243) ≤ 4eps()

    μ = chebyshevlogmoments2(Float64, N)
    w = fejerweights2(μ)
    @test norm(sum(w./(x .- 3)) - π^2/12) ≤ 4eps()
end

if VERSION < v"0.7-"
    @testset "Allocation-free ID matrix-vector products" begin
        for T in (Float64, ComplexF64)
            r = rand(T)
            A = idfact([r/(i+j-1) for i in 1:200, j = 1:50])
            P = A[:P]
            k, n = size(A)

            x = rand(T, n)
            y = zeros(T, k)

            @test norm(FastTransforms.mul!(y, A, P, x, 1, 1) - A*x) < 10eps()*norm(A*x)

            x = rand(T, k)
            y = zeros(T, n)

            @test norm(FastTransforms.At_mul_B!(y, A, P, x, 1, 1) - At_mul_B(A,x), Inf) < 10eps()

            fill!(y, zero(T))
            @test norm(FastTransforms.Ac_mul_B!(y, A, P, x, 1, 1) - A'x, Inf) < 10eps()
        end
    end
else
    @testset "Allocation-free ID matrix-vector products" begin
        for T in (Float64, ComplexF64)
            r = rand(T)
            A = idfact([r/(i+j-1) for i in 1:200, j = 1:50])
            P = A[:P]
            k, n = size(A)

            x = rand(T, n)
            y = zeros(T, k)

            @test norm(FastTransforms.mul!(y, A, P, x, 1, 1) - A*x) < 10eps()*norm(A*x)

            x = rand(T, k)
            y = zeros(T, n)

            @test norm(FastTransforms.At_mul_B!(y, A, P, x, 1, 1) - transpose(A)*x, Inf) < 10eps()

            fill!(y, zero(T))
            @test norm(FastTransforms.Ac_mul_B!(y, A, P, x, 1, 1) - A'x, Inf) < 10eps()
        end
    end
end

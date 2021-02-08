using FastTransforms, LinearAlgebra, Test

import FastTransforms: pochhammer, sqrtpi, SpecialFunctions.gamma
import FastTransforms: Cnλ, Λ, lambertw, Cnαβ, Anαβ
import FastTransforms: chebyshevmoments1, chebyshevmoments2, chebyshevjacobimoments1, chebyshevjacobimoments2, chebyshevlogmoments1, chebyshevlogmoments2

@testset "Special functions" begin
    @test pochhammer(2,3) == 24
    @test pochhammer(0.5,3) == 0.5*1.5*2.5
    @test pochhammer(0.5,0.5) == 1/sqrtpi
    @test pochhammer(0,1) == 0
    @test pochhammer(-1,2) == 0
    @test pochhammer(-5,3) == -60
    @test pochhammer(-1,-0.5) == 0
    @test 1.0/pochhammer(-0.5,-0.5) == 0
    @test pochhammer(-1+0im,-1) == -0.5
    @test pochhammer(2,1) == pochhammer(2,1.0) == pochhammer(2.0,1) == 2
    @test pochhammer(1.1,2.2) ≈ gamma(3.3)/gamma(1.1)
    @test pochhammer(-2,1) == pochhammer(-2,1.0) == pochhammer(-2.0,1) == -2

    n = 0:1000
    λ = 0.125
    @test norm(Cnλ.(n, λ) ./ Cnλ.(n, big(λ)) .- 1, Inf) < 3eps()

    x = range(0, stop=20, length=81)
    @test norm((Λ.(x) .- Λ.(big.(x)))./Λ.(x), Inf) < 2eps()
    @test norm((lambertw.(x) .- lambertw.(big.(x)))./max.(lambertw.(x), 1), Inf) < 2eps()

    x = 0:0.5:1000
    λ₁, λ₂ = 0.125, 0.875
    @test norm((Λ.(x,λ₁,λ₂) .- Λ.(big.(x),big(λ₁),big(λ₂)))./Λ.(big.(x),big(λ₁),big(λ₂)), Inf) < 4eps()
    λ₁, λ₂ = 1//3, 2//3
    @test norm((Λ.(x,Float64(λ₁),Float64(λ₂)) .- Λ.(big.(x),big(λ₁),big(λ₂))) ./ Λ.(big.(x),big(λ₁),big(λ₂)), Inf) < 4eps()

    α, β = 0.125, 0.375

    @test norm(Cnαβ.(n,α,β) ./ Cnαβ.(n,big(α),big(β)) .- 1, Inf) < 3eps()
    @test norm(Anαβ.(n,α,β) ./ Anαβ.(n,big(α),big(β)) .- 1, Inf) < 4eps()
end

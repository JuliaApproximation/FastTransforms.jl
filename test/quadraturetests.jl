using FastTransforms, LinearAlgebra, Test

import FastTransforms: chebyshevmoments1, chebyshevmoments2,
                       chebyshevjacobimoments1, chebyshevjacobimoments2,
                       chebyshevlogmoments1, chebyshevlogmoments2

@testset "Fejér and Clenshaw–Curtis quadrature" begin
    N = 20
    f = x -> exp(x)

    x = clenshawcurtisnodes(Float64, N)
    μ = chebyshevmoments1(Float64, N)
    w = clenshawcurtisweights(μ)
    @test norm(dot(f.(x), w)-2sinh(1)) ≤ 4eps()

    μ = chebyshevjacobimoments1(Float64, N, 0.25, 0.35)
    w = clenshawcurtisweights(μ)
    @test norm(dot(f.(x), w)-2.0351088204147243) ≤ 4eps()

    μ = chebyshevlogmoments1(Float64, N)
    w = clenshawcurtisweights(μ)
    @test norm(sum(w./(x .- 3)) - π^2/12) ≤ 4eps()

    x = fejernodes1(Float64, N)
    μ = chebyshevmoments1(Float64, N)
    w = fejerweights1(μ)
    @test norm(dot(f.(x), w)-2sinh(1)) ≤ 4eps()

    μ = chebyshevjacobimoments1(Float64, N, 0.25, 0.35)
    w = fejerweights1(μ)
    @test norm(dot(f.(x), w)-2.0351088204147243) ≤ 4eps()

    μ = chebyshevlogmoments1(Float64, N)
    w = fejerweights1(μ)
    @test norm(sum(w./(x .- 3)) - π^2/12) ≤ 4eps()

    x = fejernodes2(Float64, N)
    μ = chebyshevmoments2(Float64, N)
    w = fejerweights2(μ)
    @test norm(dot(f.(x), w)-2sinh(1)) ≤ 4eps()

    μ = chebyshevjacobimoments2(Float64, N, 0.25, 0.35)
    w = fejerweights2(μ)
    @test norm(dot(f.(x), w)-2.0351088204147243) ≤ 4eps()

    μ = chebyshevlogmoments2(Float64, N)
    w = fejerweights2(μ)
    @test norm(sum(w./(x .- 3)) - π^2/12) ≤ 4eps()
end

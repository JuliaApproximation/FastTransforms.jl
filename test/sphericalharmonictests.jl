using FastTransforms, Base.Test

srand(0)

println()
println("Testing slow plan")
println()

include("test_slowplan.jl")

println()
println("Testing fast plan")
println()

include("test_fastplan.jl")

println()
println("Testing thin plan")
println()

include("test_thinplan.jl")

println()
println("Testing pointwise evaluation")
println()

import FastTransforms: sphevaluatepi, normalizecolumns!, maxcolnorm

n = 256
A = sphrandn(Float64, n, n);
normalizecolumns!(A);
B = sph2fourier(A);

for θ in (0.123, 0.456)
    S0 = sum(cospi((ℓ-1)*θ)*B[ℓ,1] for ℓ in 1:n)
    SA = sum(sphevaluatepi(θ,ℓ-1,0)*A[ℓ,1] for ℓ in 1:n)
    @test norm(S0-SA) < 1000eps()
    for m in 3:2:n
        S0 = sum(cospi((ℓ-1)*θ)*B[ℓ,2m-2] for ℓ in 1:n)
        SA = sum(sphevaluatepi(θ,ℓ+m-2,m-1)*A[ℓ,2m-2] for ℓ in 1:n)
        @test norm(S0-SA) < 1000eps()
        S0 = sum(cospi((ℓ-1)*θ)*B[ℓ,2m-1] for ℓ in 1:n)
        SA = sum(sphevaluatepi(θ,ℓ+m-2,m-1)*A[ℓ,2m-1] for ℓ in 1:n)
        @test norm(S0-SA) < 1000eps()
    end
end

for θ in (0.123, 0.456), m in 2:2:n
    S1 = sum(sinpi(ℓ*θ)*B[ℓ,2m-2] for ℓ in 1:n)
    SA = sum(sphevaluatepi(θ,ℓ+m-2,m-1)*A[ℓ,2m-2] for ℓ in 1:n)
    @test norm(S1-SA) < 1000eps()
    S1 = sum(sinpi(ℓ*θ)*B[ℓ,2m-1] for ℓ in 1:n)
    SA = sum(sphevaluatepi(θ,ℓ+m-2,m-1)*A[ℓ,2m-1] for ℓ in 1:n)
    @test norm(S1-SA) < 1000eps()
end

println()
println("Testing API")
println()

import FastTransforms: normalizecolumns!, maxcolnorm

n = VERSION < v"0.6.0-" ? 256 : 512
A = sphrandn(Float64, n, n);
normalizecolumns!(A);

B = sph2fourier(A)
C = fourier2sph(B)
println("The backward difference between slow plan and original: ", maxcolnorm(A-C))

P = plan_sph2fourier(A)
B = P*A
C = P\B

println("The backward difference between slow plan and original: ", maxcolnorm(A-C))

if VERSION ≥ v"0.6.0-"
    n = 1024
    A = sphrandn(Float64, n, n);
    normalizecolumns!(A);

    B = sph2fourier(A; sketch = :none)
    C = fourier2sph(B; sketch = :none)
    println("The backward difference between thin plan and original: ", maxcolnorm(A-C))

    P = plan_sph2fourier(A; sketch = :none)
    B = P*A
    C = P\B

    println("The backward difference between thin plan and original: ", maxcolnorm(A-C))
end

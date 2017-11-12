using FastTransforms, Base.Test, Compat
using Compat.Test

import FastTransforms: normalizecolumns!, maxcolnorm, sphevaluatepi

@testset "Pointwise evaluation" begin
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
end

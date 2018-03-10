using FastTransforms, Compat
using Compat.Test

import FastTransforms: normalizecolumns!, maxcolnorm

@testset "Synthesis and analysis" begin

    # Starting with normalized spherical harmonic coefficients,

    n = 50
    F = sphrandn(Float64, n, n);
    normalizecolumns!(F);

    # we convert to bivariate Fourier series.

    G = sph2fourier(F);

    # At equispaced points in angle,

    θ = (0.5:n-0.5)*π/n
    φ = (0:2n-2)*2π/(2n-1)

    # the Fourier series evaluates to:

    SF = [sum(G[ℓ,1]*cos((ℓ-1)*θ)/sqrt(2π) for ℓ in 1:n) + sum(G[ℓ,2m]*sin(ℓ*θ)*sin(m*φ)/sqrt(π) for ℓ in 1:n, m in 1:2:n-1) + sum(G[ℓ,2m+1]*sin(ℓ*θ)*cos(m*φ)/sqrt(π) for ℓ in 1:n, m in 1:2:n-1) + sum(G[ℓ,2m]*cos((ℓ-1)*θ)*sin(m*φ)/sqrt(π) for ℓ in 1:n, m in 2:2:n-1) + sum(G[ℓ,2m+1]*cos((ℓ-1)*θ)*cos(m*φ)/sqrt(π) for ℓ in 1:n, m in 2:2:n-1) for θ in θ, φ in φ]

    # but that was slow, so we accelerate it via in-place FFTW technology:

    Ps = FastTransforms.plan_synthesis(G);

    Y = zero(G);

    A_mul_B!(Y, Ps, G)

    @test maxcolnorm(SF - Y) < 10000eps()

    # Retracing our steps, function values on the sphere are converted to Fourier coefficients:

    Pa = FastTransforms.plan_analysis(Y);

    Z = zero(Y);

    A_mul_B!(Z, Pa, Y)

    @test maxcolnorm(Z - G) < 10eps()

    # And Fourier coefficients are converted back to spherical harmonic coefficients:

    H = fourier2sph(Z)

    @test maxcolnorm(F - H) < 100eps()
end

@testset "Test for dropping last column" begin
    for f in ((θ,φ)->1/(3 + cospi(φ) + sinpi(θ)),
              (θ,φ)->cos(50*cospi(φ)*sinpi(θ)*sinpi(φ)*sinpi(θ)),
              (θ,φ)->cos(50*cospi(φ)*sinpi(θ)+80*sinpi(φ)*sinpi(θ)),
              (θ,φ)->sqrt(5+cospi(φ)*sinpi(θ)+exp(sinpi(φ)*sinpi(θ))+sin(cospi(θ))))
        n = 20
        UO = sphones(Float64, n, n)
        UE = sphones(Float64, n, n)
        while norm(UO[:,end]) > vecnorm(UO)*eps()
            θ = (0.5:n-0.5)/n
            φ = (0:2n-2)*2/(2n-1)
            F = [f(θ,φ) for θ in θ, φ in φ]
            V = zero(F)
            A_mul_B!(V, FastTransforms.plan_analysis(F), F)
            UO = fourier2sph(V)

            φ = (0:2n-3)*2/(2n-2)
            F = [f(θ,φ) for θ in θ, φ in φ]
            V = zero(F)
            A_mul_B!(V, FastTransforms.plan_analysis(F), F)
            UE = fourier2sph(V)

            n *= 2
        end
        @test vecnorm(UO[:,1:end-1] - UE) < n*vecnorm(UO)*eps()
    end
end

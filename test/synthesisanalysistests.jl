using FastTransforms, Base.Test

import FastTransforms: normalizecolumns!, maxcolnorm

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

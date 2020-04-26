#############
# This example plays with analysis of:
#
#   f(r) = e^{i k⋅r},
#
# for some k ∈ ℝ³ and where r ∈ 𝕊², using spin-0 spherical harmonics.
#
# It applies ð, the spin-raising operator,
# both on the spin-0 coefficients as well as the original function,
# followed by a spin-1 analysis to compare coefficients.
#
# See also sphere.jl
# For the storage pattern of the arrays, please consult the documentation.
#############

using FastTransforms, LinearAlgebra

# The colatitudinal grid (mod π):
N = 10
θ = (0.5:N-0.5)/N

# The longitudinal grid (mod π):
M = 2*N-1
φ = (0:M-1)*2/M

k = [2/7, 3/7, 6/7]
r = (θ,φ) -> [sinpi(θ)*cospi(φ), sinpi(θ)*sinpi(φ), cospi(θ)]

# On the tensor product grid, our function samples are:

F = [exp(im*(k⋅r(θ,φ))) for θ in θ, φ in φ]

P = plan_spinsph2fourier(F, 0)
PA = plan_spinsph_analysis(F, 0)

# Its spin-0 spherical harmonic coefficients are:

U⁰ = P\(PA*F)

norm(U⁰) ≈ sqrt(4π)

# Spin can be incremented by applying ð, either on the spin-0 coefficients:

U¹c = zero(U⁰)
for n in 1:N-1
    U¹c[n, 1] = sqrt(n*(n+1))*U⁰[n+1, 1]
end
for m in 1:M÷2
    for n in 0:N-1
        U¹c[n+1, 2m] = -sqrt((n+m)*(n+m+1))*U⁰[n+1, 2m]
        U¹c[n+1, 2m+1] = sqrt((n+m)*(n+m+1))*U⁰[n+1, 2m+1]
    end
end

# or on the original function through analysis with spin-1 spherical harmonics:

F = [-(k[1]*(im*cospi(θ)*cospi(φ) + sinpi(φ)) + k[2]*(im*cospi(θ)*sinpi(φ)-cospi(φ)) - im*k[3]*sinpi(θ))*exp(im*(k⋅r(θ,φ))) for θ in θ, φ in φ]

P = plan_spinsph2fourier(F, 1)
PA = plan_spinsph_analysis(F, 1)

U¹s = P\(PA*F)

norm(U¹c) ≈ norm(U¹s) ≈ sqrt(8π/3*(k⋅k))

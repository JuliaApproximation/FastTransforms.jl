#############
# This example confirms numerically that
#
#   [P₄(z⋅y) - P₄(x⋅y)]/(z⋅y - x⋅y),
#
# is actually a degree-3 polynomial on 𝕊², where P₄ is the degree-4
# Legendre polynomial, and x,y,z ∈ 𝕊².
# To verify, we sample the function on a 5×9 tensor product grid
# at equispaced points-in-angle defined by:
#
#   θₙ = (n+1/2)π/N, for 0 ≤ n < N, and
#
#   φₘ = 2π m/M, for 0 ≤ m < M;
#
# we convert the function samples to Fourier coefficients using
# `plan_sph_analysis`; and finally, we transform
# the Fourier coefficients to spherical harmonic coefficients using
# `plan_sph2fourier`.
#
# In the basis of spherical harmonics, it is plain to see the
# addition theorem in action, since P₄(x⋅y) should only consist of
# exact-degree-4 harmonics.
#
# For the storage pattern of the arrays, please consult the documentation.
#############

function threshold!(A::AbstractArray, ϵ)
    for i in eachindex(A)
        if abs(A[i]) < ϵ A[i] = 0 end
    end
    A
end

using FastTransforms, LinearAlgebra

# The colatitudinal grid (mod π):
N = 5
θ = (0.5:N-0.5)/N

# The longitudinal grid (mod π):
M = 2*N-1
φ = (0:M-1)*2/M

# Arbitrarily, we place x at the North pole:
x = [0,0,1]

# Another vector is completely free:
y = normalize([.123,.456,.789])

# Thus z ∈ 𝕊² is our variable vector, parameterized in spherical coordinates:
z = (θ,φ) -> [sinpi(θ)*cospi(φ), sinpi(θ)*sinpi(φ), cospi(θ)]

# The degree-4 Legendre polynomial is:
P4 = x -> (35*x^4-30*x^2+3)/8

# On the tensor product grid, our function samples are:
F = [(P4(z(θ,φ)⋅y) - P4(x⋅y))/(z(θ,φ)⋅y - x⋅y) for θ in θ, φ in φ]

P = plan_sph2fourier(F)
PA = plan_sph_analysis(F)

# Its spherical harmonic coefficients demonstrate that it is degree-3:
V = PA*F
U3 = threshold!(P\V, 400*eps())

# Similarly, on the tensor product grid, the Legendre polynomial P₄(z⋅y) is:
F = [P4(z(θ,φ)⋅y) for θ in θ, φ in φ]

# Its spherical harmonic coefficients demonstrate that it is exact-degree-4:
V = PA*F
U4 = threshold!(P\V, 3*eps())

nrm1 = norm(U4);

# Finally, the Legendre polynomial P₄(z⋅x) is aligned with the grid:
F = [P4(z(θ,φ)⋅x) for θ in θ, φ in φ]

# It only has one nonnegligible spherical harmonic coefficient.
# Can you spot it?
V = PA*F
U4 = threshold!(P\V, 3*eps())

# That nonnegligible coefficient should be approximately √(2π/(4+1/2)),
# since the convention in this library is to orthonormalize.

nrm2 = norm(U4);

# Note that the integrals of both functions P₄(z⋅y) and P₄(z⋅x) and their
# L²(𝕊²) norms are the same because of rotational invariance. The integral of
# either is perhaps not interesting as it is mathematically zero, but the norms
# of either should be approximately the same.

@show nrm1
@show nrm2
@show nrm1 ≈ nrm2

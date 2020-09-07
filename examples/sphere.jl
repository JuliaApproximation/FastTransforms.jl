# ## Spherical harmonic addition theorem
# This example confirms numerically that
# ```math
# \frac{P_4(z\cdot y) - P_4(x\cdot y)}{z\cdot y - x\cdot y},
# ```
#
# is actually a degree-$3$ polynomial on $\mathbb{S}^2$, where $P_4$ is the degree-$4$
# Legendre polynomial, and $x,y,z \in \mathbb{S}^2$.
# To verify, we sample the function on a $5\times9$ equiangular grid
# defined by:
# ```math
# \theta_n = (n+\frac{1}{2})\pi/N,\quad{\rm for}\quad 0\le n < N,\quad{\rm and}
# ```
# ```math
# \varphi_m = 2\pi m/M,\quad{\rm for}\quad 0\le m < M;
# ```
# we convert the function samples to Fourier coefficients using
# `plan_sph_analysis`; and finally, we transform
# the Fourier coefficients to spherical harmonic coefficients using
# `plan_sph2fourier`.
#
# In the basis of spherical harmonics, it is plain to see the
# addition theorem in action, since $P_4(x\cdot y)$ should only consist of
# exact-degree-$4$ harmonics.
#
# For the storage pattern of the arrays, please consult the
# [documentation](https://MikaelSlevinsky.github.io/FastTransforms).

function threshold!(A::AbstractArray, ϵ)
    for i in eachindex(A)
        if abs(A[i]) < ϵ A[i] = 0 end
    end
    A
end

using FastTransforms, LinearAlgebra

# The colatitudinal grid (mod $\pi$):
N = 5
θ = (0.5:N-0.5)/N

# The longitudinal grid (mod $\pi$):
M = 2*N-1
φ = (0:M-1)*2/M

# Arbitrarily, we place $x$ at the North pole:
x = [0,0,1]

# Another vector is completely free:
y = normalize([.123,.456,.789])

# Thus $z \in \mathbb{S}^2$ is our variable vector, parameterized in spherical coordinates:
z = (θ,φ) -> [sinpi(θ)*cospi(φ), sinpi(θ)*sinpi(φ), cospi(θ)]

# The degree-$4$ Legendre polynomial is:
P4 = x -> (35*x^4-30*x^2+3)/8

# On the tensor product grid, our function samples are:
F = [(P4(z(θ,φ)⋅y) - P4(x⋅y))/(z(θ,φ)⋅y - x⋅y) for θ in θ, φ in φ]

# We precompute a spherical harmonic--Fourier plan:
P = plan_sph2fourier(F)

# And an FFTW Fourier analysis plan on $\mathbb{S}^2$:
PA = plan_sph_analysis(F)

# Its spherical harmonic coefficients demonstrate that it is degree-$3$:
V = PA*F
U3 = threshold!(P\V, 400*eps())

# Similarly, on the tensor product grid, the Legendre polynomial $P_4(z\cdot y)$ is:
F = [P4(z(θ,φ)⋅y) for θ in θ, φ in φ]

# Its spherical harmonic coefficients demonstrate that it is exact-degree-$4$:
V = PA*F
U4 = threshold!(P\V, 3*eps())

# The $L^2(\mathbb{S}^2)$ norm of the function is:
nrm1 = norm(U4)

# Finally, the Legendre polynomial $P_4(z\cdot x)$ is aligned with the grid:
F = [P4(z(θ,φ)⋅x) for θ in θ, φ in φ]

# It only has one nonnegligible spherical harmonic coefficient.
# Can you spot it?
V = PA*F
U4 = threshold!(P\V, 3*eps())

# That nonnegligible coefficient should be approximately `√(2π/(4+1/2))`,
# since the convention in this library is to orthonormalize.

nrm2 = norm(U4)

# Note that the integrals of both functions $P_4(z\cdot y)$ and $P_4(z\cdot x)$ and their
# $L^2(\mathbb{S}^2)$ norms are the same because of rotational invariance. The integral of
# either is perhaps not interesting as it is mathematically zero, but the norms
# of either should be approximately the same.

nrm1 ≈ nrm2

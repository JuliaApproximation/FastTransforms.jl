# # Spherical harmonic addition theorem
# This example confirms numerically that
# ```math
# f(z) = \frac{P_n(z\cdot y) - P_n(x\cdot y)}{z\cdot y - x\cdot y},
# ```
# is actually a degree-$(n-1)$ polynomial on $\mathbb{S}^2$, where $P_n$ is the degree-$n$
# Legendre polynomial, and $x,y,z \in \mathbb{S}^2$.
# To verify, we sample the function on a $N\times M$ equiangular grid
# defined by:
# ```math
# \begin{aligned}
# \theta_n & = (n+\tfrac{1}{2})\pi/N,\quad{\rm for}\quad 0\le n < N,\quad{\rm and}\\
# \varphi_m & = 2\pi m/M,\quad{\rm for}\quad 0\le m < M;
# \end{aligned}
# ```
# we convert the function samples to Fourier coefficients using
# `plan_sph_analysis`; and finally, we transform
# the Fourier coefficients to spherical harmonic coefficients using
# `plan_sph2fourier`.
#
# In the basis of spherical harmonics, it is plain to see the
# addition theorem in action, since $P_n(x\cdot y)$ should only consist of
# exact-degree-$n$ harmonics.
#
# For the storage pattern of the arrays, please consult the
# [documentation](https://MikaelSlevinsky.github.io/FastTransforms).

function threshold!(A::AbstractArray, ϵ)
    for i in eachindex(A)
        if abs(A[i]) < ϵ A[i] = 0 end
    end
    A
end

using FastTransforms, LinearAlgebra, Plots
const GENFIGS = joinpath(dirname(dirname(pathof(FastTransforms))), "docs/src/generated")
!isdir(GENFIGS) && mkdir(GENFIGS)
plotlyjs()

# The colatitudinal grid (mod $\pi$):
N = 15
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

# On the tensor product grid, the Legendre polynomial $P_n(z\cdot y)$ is:
A = [(2k+1)/(k+1) for k in 0:N-1]
B = zeros(N)
C = [k/(k+1) for k in 0:N]
c = zeros(N); c[N] = 1
pts = vec([z(θ, φ)⋅y for θ in θ, φ in φ])
phi0 = ones(N*M)
F = reshape(FastTransforms.clenshaw!(c, A, B, C, pts, phi0, zeros(N*M)), N, M)

# We superpose a surface plot of $f$ on top of the grid:
X = [sinpi(θ)*cospi(φ) for θ in θ, φ in φ]
Y = [sinpi(θ)*sinpi(φ) for θ in θ, φ in φ]
Z = [cospi(θ) for θ in θ, φ in φ]
scatter3d(vec(X), vec(Y), vec(Z); markersize=1.25, markercolor=:violetred)
surface!(X, Y, Z; surfacecolor=F, legend=false, xlabel="x", ylabel="y", zlabel="f")
savefig(joinpath(GENFIGS, "sphere1.html"))
###```@raw html
###<object type="text/html" data="../sphere1.html" style="width:100%;height:400px;"></object>
###```

# We show the cut in the surface to help illustrate the definition of the grid.
# In particular, we do not sample the poles.
#
# We precompute a spherical harmonic--Fourier plan:
P = plan_sph2fourier(F)

# And an FFTW Fourier analysis plan on $\mathbb{S}^2$:
PA = plan_sph_analysis(F)

# Its spherical harmonic coefficients demonstrate that it is exact-degree-$n$:
V = PA*F
U = threshold!(P\V, 400*eps())

# The $L^2(\mathbb{S}^2)$ norm of the function is:
nrm1 = norm(U)

# Similarly, on the tensor product grid, our function samples are:
Pnxy = FastTransforms.clenshaw!(c, A, B, C, [x⋅y], [1.0], [0.0])[1]
F = [(F[n, m] - Pnxy)/(z(θ[n], φ[m])⋅y - x⋅y) for n in 1:N, m in 1:M]

# We superpose a surface plot of $f$ on top of the grid:
scatter3d(vec(X), vec(Y), vec(Z); markersize=1.25, markercolor=:violetred)
surface!(X, Y, Z; surfacecolor=F, legend=false, xlabel="x", ylabel="y", zlabel="f")
savefig(joinpath(GENFIGS, "sphere2.html"))
###```@raw html
###<object type="text/html" data="../sphere2.html" style="width:100%;height:400px;"></object>
###```

# Its spherical harmonic coefficients demonstrate that it is degree-$(n-1)$:
V = PA*F
U = threshold!(P\V, 400*eps())

# Finally, the Legendre polynomial $P_n(z\cdot x)$ is aligned with the grid:
pts = vec([z(θ, φ)⋅x for θ in θ, φ in φ])
F = reshape(FastTransforms.clenshaw!(c, A, B, C, pts, phi0, zeros(N*M)), N, M)

# We superpose a surface plot of $f$ on top of the grid:
scatter3d(vec(X), vec(Y), vec(Z); markersize=1.25, markercolor=:violetred)
surface!(X, Y, Z; surfacecolor=F, legend=false, xlabel="x", ylabel="y", zlabel="f")
savefig(joinpath(GENFIGS, "sphere3.html"))
###```@raw html
###<object type="text/html" data="../sphere3.html" style="width:100%;height:400px;"></object>
###```

# It only has one nonnegligible spherical harmonic coefficient.
# Can you spot it?
V = PA*F
U = threshold!(P\V, 400*eps())

# That nonnegligible coefficient should be
ret = eval("√(2π/($(N-1)+1/2))")

# which is approximately
eval(Meta.parse(ret))

# since the convention in this library is to orthonormalize.
nrm2 = norm(U)

# Note that the integrals of both functions $P_n(z\cdot y)$ and $P_n(z\cdot x)$ and their
# $L^2(\mathbb{S}^2)$ norms are the same because of rotational invariance. The integral of
# either is perhaps not interesting as it is mathematically zero, but the norms
# of either should be approximately the same.
nrm1 ≈ nrm2

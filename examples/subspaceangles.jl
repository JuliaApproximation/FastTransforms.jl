# # Subspace angles
# This example considers the angles between neighbouring Laguerre polynomials with a perturbed measure:
# ```math
# \cos\theta_n = \frac{\langle L_n, L_{n+k}\rangle}{\|L_n\|_2 \|L_{n+k}\|_2},\quad{\rm for}\quad 0\le n < N-k,
# ```
# where the inner product is defined by $\langle f, g\rangle = \int_0^\infty f(x) g(x) x^\beta e^{-x}{\rm\,d}x$.
#
# We do so by connecting Laguerre polynomials to the normalized generalized Laguerre polynomials associated with the perturbed measure. It follows by the inner product of the connection coefficients that:
# ```math
# \cos\theta_n = \frac{(V^\top V)_{n, n+k}}{\sqrt{(V^\top V)_{n, n}(V^\top V)_{n+k, n+k}}}.
# ```
#
using FastTransforms, LinearAlgebra

# The neighbouring index `k` and the maximum degree `N-1`:
k, N = 1, 11

# The Laguerre connection parameters:
α, β = 0.0, 0.125

# We precompute a Laguerre--Laguerre plan:
P = plan_lag2lag(Float64, N, α, β; norm2=true)

# We apply the plan to the identity, followed by the adjoint plan:
VtV = parent(P*I)
lmul!(P', VtV)

# From this matrix, the angles are recovered from:
θ = [acos(VtV[n, n+k]/sqrt(VtV[n, n]*VtV[n+k, n+k])) for n in 1:N-k]

# # Automatic differentiation through spherical harmonic transforms
# This example finds a positive value of $\lambda$ in:
# ```math
# f(r) = \sin[\lambda (k\cdot r)],
# ```
# for some $k,r\in\mathbb{S}^2$ such that $\int_{\mathbb{S}^2} f^2 {\rm\,d}\Omega = 1$.
# We do this by using derivative information through:
# ```math
# \dfrac{\partial f}{\partial \lambda} = (k\cdot r) \cos[\lambda (k\cdot r)].
# ```

using FastTransforms, LinearAlgebra

# The colatitudinal grid (mod $\pi$):
N = 15
θ = (0.5:N-0.5)/N

# The longitudinal grid (mod $\pi$):
M = 2*N-1
φ = (0:M-1)*2/M

# We precompute a spherical harmonic--Fourier plan:
P = plan_sph2fourier(Float64, N)

# And an FFTW Fourier analysis plan on $\mathbb{S}^2$:
PA = plan_sph_analysis(Float64, N, M)

# Our choice of $k$ and angular parametrization of $r$:
k = [2/7, 3/7, 6/7]
r = (θ,φ) -> [sinpi(θ)*cospi(φ), sinpi(θ)*sinpi(φ), cospi(θ)]

# Our initial guess for $\lambda$:
λ = 1.0

# Then we run Newton iteration and grab an espresso:
for _ in 1:7
    F = [sin(λ*(k⋅r(θ,φ))) for θ in θ, φ in φ]
    Fλ = [(k⋅r(θ,φ))*cos(λ*(k⋅r(θ,φ))) for θ in θ, φ in φ]
    U = P\(PA*F)
    Uλ = P\(PA*Fλ)
    global λ = λ - (norm(U)^2-1)/(2*sum(U.*Uλ))
    println("λ: $(rpad(λ, 18)) and the 2-norm: $(rpad(norm(U), 18))")
end

# # Integration on an annulus
# In this example, we explore integration of the function:
# ```math
#   f(x,y) = \frac{x^3}{x^2+y^2-\frac{1}{4}},
# ```
# over the annulus defined by $\{(r,\theta) : \frac{2}{3} < r < 1, 0 < \theta < 2\pi\}$.
# We will calculate the integral:
# ```math
#   \int_0^{2\pi}\int_{\frac{2}{3}}^1 f(r\cos\theta,r\sin\theta)^2r{\rm\,d}r{\rm\,d}\theta,
# ```
# by analyzing the function in an annulus polynomial series.
# We analyze the function on an $N\times M$ tensor product grid defined by:
# ```math
# \begin{aligned}
# r_n & = \sqrt{\cos^2\left[(n+\tfrac{1}{2})\pi/2N\right] + \rho^2 \sin^2\left[(n+\tfrac{1}{2})\pi/2N\right]},\quad{\rm for}\quad 0\le n < N,\quad{\rm and}\\
# \theta_m & = 2\pi m/M,\quad{\rm for}\quad 0\le m < M;
# \end{aligned}
# ```
# we convert the function samples to Chebyshev×Fourier coefficients using
# `plan_annulus_analysis`; and finally, we transform the Chebyshev×Fourier
# coefficients to annulus polynomial coefficients using `plan_ann2cxf`.
#
# For the storage pattern of the arrays, please consult the
# [documentation](https://MikaelSlevinsky.github.io/FastTransforms).

using FastTransforms, LinearAlgebra, Plots
const GENFIGS = joinpath(pkgdir(FastTransforms), "docs/src/generated")
!isdir(GENFIGS) && mkdir(GENFIGS)
plotlyjs()

# Our function $f$ on the annulus:
f = (x,y) -> x^3/(x^2+y^2-1/4)

# The annulus polynomial degree:
N = 8
M = 4N-3

# The annulus inner radius:
ρ = 2/3

# The radial grid:
r = [begin t = (N-n-0.5)/(2N); ct2 = sinpi(t); st2 = cospi(t); sqrt(ct2^2+ρ^2*st2^2) end; for n in 0:N-1]

# The angular grid (mod $\pi$):
θ = (0:M-1)*2/M

# On the mapped tensor product grid, our function samples are:
F = [f(r*cospi(θ), r*sinpi(θ)) for r in r, θ in θ]

# We superpose a surface plot of $f$ on top of the grid:
X = [r*cospi(θ) for r in r, θ in θ]
Y = [r*sinpi(θ) for r in r, θ in θ]
scatter3d(vec(X), vec(Y), vec(0F); markersize=0.75, markercolor=:red)
surface!(X, Y, F; legend=false, xlabel="x", ylabel="y", zlabel="f")
savefig(joinpath(GENFIGS, "annulus.html"))
###```@raw html
###<object type="text/html" data="../annulus.html" style="width:100%;height:400px;"></object>
###```

# We precompute an Annulus--Chebyshev×Fourier plan:
α, β, γ = 0, 0, 0
P = plan_ann2cxf(F, α, β, γ, ρ)

# And an FFTW Chebyshev×Fourier analysis plan on the annulus:
PA = plan_annulus_analysis(F, ρ)

# Its annulus coefficients are:
U = P\(PA*F)

# The annulus coefficients are useful for integration.
# The integral of $[f(x,y)]^2$ over the annulus is
# approximately the square of the 2-norm of the coefficients:
norm(U)^2, 5π/8*(1675/4536+9*log(3)/32-3*log(7)/32)

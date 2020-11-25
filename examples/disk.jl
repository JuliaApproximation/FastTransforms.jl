# # Holomorphic integration on the unit disk
# In this example, we explore integration of a harmonic function:
# ```math
#   f(x,y) = \frac{x^2-y^2+1}{(x^2-y^2+1)^2+(2xy+1)^2},
# ```
# over the unit disk. In this case, we know from complex analysis that the
# integral of a holomorphic function is equal to $\pi \times f(0,0)$.
# We analyze the function on an $N\times M$ tensor product grid defined by:
# ```math
# \begin{aligned}
# r_n & = \cos\left[(n+\tfrac{1}{2})\pi/2N\right],\quad{\rm for}\quad 0\le n < N,\quad{\rm and}\\
# \theta_m & = 2\pi m/M,\quad{\rm for}\quad 0\le m < M;
# \end{aligned}
# ```
# we convert the function samples to Chebyshev×Fourier coefficients using
# `plan_disk_analysis`; and finally, we transform the Chebyshev×Fourier
# coefficients to Zernike polynomial coefficients using `plan_disk2cxf`.
#
# For the storage pattern of the arrays, please consult the
# [documentation](https://MikaelSlevinsky.github.io/FastTransforms).

using FastTransforms, LinearAlgebra, Plots
plotlyjs()

# Our function $f$ on the disk:
f = (x,y) -> (x^2-y^2+1)/((x^2-y^2+1)^2+(2x*y+1)^2)

# The Zernike polynomial degree:
N = 20
M = 4N-3

# The radial grid:
r = [sinpi((N-n-0.5)/(2N)) for n in 0:N-1]

# The angular grid (mod $\pi$):
θ = (0:M-1)*2/M

# On the mapped tensor product grid, our function samples are:
F = [f(r*cospi(θ), r*sinpi(θ)) for r in r, θ in θ]

# We superpose a surface plot of $f$ on top of the grid:
X = [r*cospi(θ) for r in r, θ in θ]
Y = [r*sinpi(θ) for r in r, θ in θ]
scatter3d(vec(X), vec(Y), vec(0X); markersize=0.75, markercolor=:red, size=(1000, 1000))
surface!(X, Y, F; legend=false, xlabel="x", ylabel="y", zlabel="f")

# We precompute a (generalized) Zernike--Chebyshev×Fourier plan:
α, β = 0, 0
P = plan_disk2cxf(F, α, β)

# And an FFTW Chebyshev×Fourier analysis plan on the disk:
PA = plan_disk_analysis(F)

# Its Zernike coefficients are:
U = P\(PA*F)

# The Zernike coefficients are useful for integration. The integral of $f(x,y)$
# over the disk should be $\pi/2$ by harmonicity. The coefficient of $Z_{0,0}$
# multiplied by `√π` is:
U[1, 1]*sqrt(π)

# Using an orthonormal basis, the integral of $[f(x,y)]^2$ over the disk is
# approximately the square of the 2-norm of the coefficients:
norm(U)^2, π/(2*sqrt(2))*log1p(sqrt(2))

# But there's more! Next, we repeat the experiment using the Dunkl-Xu
# orthonormal polynomials supported on the rectangularized disk.
N = 2N
M = N

# We analyze the function on an $N\times M$ mapped tensor product $xy$-grid defined by:
# ```math
# \begin{aligned}
# x_n & = \cos\left(\frac{2n+1}{2N}\pi\right) = \sin\left(\frac{N-2n-1}{2N}\pi\right),\quad {\rm for} \quad 0 \le n < N,\quad{\rm and}\\
# z_m & = \cos\left(\frac{2m+1}{2M}\pi\right) = \sin\left(\frac{M-2m-1}{2M}\pi\right),\quad {\rm for} \quad 0 \le m < M,\\
# y_{n,m} & = \sqrt{1-x_n^2}z_m.
# \end{aligned}
# ```
# Slightly more accuracy can be expected by using an auxiliary array:
# ```math
#   w_n = \sin\left(\frac{2n+1}{2N}\pi\right),\quad {\rm for} \quad 0 \le n < N,
# ```
# so that $y_{n,m} = w_nz_m$.
#
# The x grid
w = [sinpi((n+0.5)/N) for n in 0:N-1]
x = [sinpi((N-2n-1)/(2N)) for n in 0:N-1]

# The z grid
z = [sinpi((M-2m-1)/(2M)) for m in 0:M-1]

# On the mapped tensor product grid, our function samples are:
F = [f(x[n], w[n]*z) for n in 1:N, z in z]

# We superpose a surface plot of $f$ on top of the grid:
X = [x for x in x, z in z]
Y = [w*z for w in w, z in z]
scatter3d(vec(X), vec(Y), vec(0X); markersize=0.75, markercolor=:green, size=(1000, 1000))
surface!(X, Y, F; legend=false, xlabel="x", ylabel="y", zlabel="f")

# We precompute a Dunkl-Xu--Chebyshev plan:
P = plan_rectdisk2cheb(F, β)

# And an FFTW Chebyshev² analysis plan on the rectangularized disk:
PA = plan_rectdisk_analysis(F)

# Its Dunkl-Xu coefficients are:
U = P\(PA*F)

# The Dunkl-Xu coefficients are useful for integration. The integral of $f(x,y)$
# over the disk should be $\pi/2$ by harmonicity. The coefficient of $P_{0,0}$
# multiplied by `√π` is:
U[1, 1]*sqrt(π)

# Using an orthonormal basis, the integral of $[f(x,y)]^2$ over the disk is
# approximately the square of the 2-norm of the coefficients:
norm(U)^2, π/(2*sqrt(2))*log1p(sqrt(2))

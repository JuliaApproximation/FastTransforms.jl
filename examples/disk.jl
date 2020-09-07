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

using FastTransforms, LinearAlgebra

# Our function $f$ on the disk:
f = (x,y) -> (x^2-y^2+1)/((x^2-y^2+1)^2+(2x*y+1)^2)

# The Zernike polynomial degree:
N = 5
M = 4N-3

# The radial grid:
r = [sinpi((N-n-0.5)/(2N)) for n in 0:N-1]

# The angular grid (mod $\pi$):
θ = (0:M-1)*2/M

# On the mapped tensor product grid, our function samples are:
F = [f(r*cospi(θ), r*sinpi(θ)) for r in r, θ in θ]

# We precompute a Zernike--Chebyshev×Fourier plan:
P = plan_disk2cxf(F)

# And an FFTW Chebyshev×Fourier analysis plan on the disk:
PA = plan_disk_analysis(F)

# Its Zernike coefficients are:
U = P\(PA*F)

# The Zernike coefficients are useful for integration. The integral of $f(x,y)$
# over the disk should be $\pi/2$ by harmonicity. The coefficient of $Z_0^0$
# multiplied by `√π` is:
U[1, 1]*sqrt(π)

# Using an orthonormal basis, the integral of $[f(x,y)]^2$ over the disk is
# approximately the square of the 2-norm of the coefficients:
norm(U)^2

#############
# In this example, we explore integration of a harmonic function:
#
#   f(x,y) = (x^2-y^2+1)/[(x^2-y^2+1)^2+(2xy+1)^2],
#
# over the unit disk. In this case, we know from complex analysis that the
# integral of a holomorphic function is equal to π × f(0,0).
# We analyze the function on an N×M tensor product grid defined by:
#
#   rₙ = cos[(n+1/2)π/2N], for 0 ≤ n < N, and
#
#   θₘ = 2π m/M, for 0 ≤ m < M;
#
# we convert the function samples to Chebyshev×Fourier coefficients using
# `plan_disk_analysis`; and finally, we transform the Chebyshev×Fourier
# coefficients to disk harmonic coefficients using `plan_disk2cxf`.
#
# For the storage pattern of the arrays, please consult the documentation.
#############

using FastTransforms, LinearAlgebra

f = (x,y) -> (x^2-y^2+1)/((x^2-y^2+1)^2+(2x*y+1)^2)

N = 15
M = 4N-3

r = [sinpi((N-n-0.5)/(2N)) for n in 0:N-1]
θ = 2*(0:M-1)/M # mod π.

P = plan_disk2cxf(Float64, N)
PA = plan_disk_analysis(Float64, N, M)

# On the mapped tensor product grid, our function samples are:
F = [f(r*cospi(θ), r*sinpi(θ)) for r in r, θ in θ]

# Its Zernike coefficients are:
U = P\(PA*F)

# The Zernike coefficients are useful for integration. The integral of f(x,y)
# over the disk should be π/2 by harmonicity. The coefficient of Z_0^0
# multiplied by √π is:
U[1, 1]*sqrt(π)

# Using an orthonormal basis, the integral of [f(x,y)]^2 over the disk is
# approximately the square of the 2-norm of the coefficients:
norm(U)^2

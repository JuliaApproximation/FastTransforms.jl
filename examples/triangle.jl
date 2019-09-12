#############
# In this example, we sample a bivariate function:
#
#   f(x,y) = 1/(1+x^2+y^2),
#
# on the reference triangle with vertices (0,0), (0,1), and (1,0) and analyze it
# in a Proriol series. Then, we find Proriol series for each component of its
# gradient by term-by-term differentiation of our expansion, and we compare them
# with the true Proriol series by sampling an exact expression for the gradient.
#
# We analyze f(x,y) on an N×M mapped tensor product grid defined by:
#
#   x = (1+u)/2, and y = (1-u)*(1+v)/4, where:
#
#   uₙ = cos[(n+1/2)π/N], for 0 ≤ n < N, and
#
#   vₘ = cos[(m+1/2)π/M], for 0 ≤ m < M;
#
# we convert the function samples to mapped Chebyshev² coefficients using
# `plan_tri_analysis`; and finally, we transform the mapped Chebyshev²
# coefficients to Proriol coefficients using `plan_tri2cheb`.
#
# For the storage pattern of the arrays, please consult the documentation.
#############

using FastTransforms, LinearAlgebra

f = (x,y) -> 1/(1+x^2+y^2)
fx = (x,y) -> -2x/(1+x^2+y^2)^2
fy = (x,y) -> -2y/(1+x^2+y^2)^2

N = 10
M = N

α, β, γ = 0, 0, 0

u = [sinpi((N-2n-1)/(2N)) for n in 0:N-1]
v = [sinpi((M-2m-1)/(2M)) for m in 0:M-1]

# Instead of using the u, v grid, we use one with more accuracy near the origin.
x = [sinpi((2N-2n-1)/(4N))^2 for n in 0:N-1]
w = [sinpi((2M-2m-1)/(4M))^2 for m in 0:M-1]

(1 .+ u)./2 ≈ x
(1 .- u).*(1 .+ v')/4 ≈ reverse(x).*w'

# On the mapped tensor product grid, our function samples are:
F = [f(x[n+1], x[N-n]*w[m+1]) for n in 0:N-1, m in 0:M-1]

P = plan_tri2cheb(F, α, β, γ)
PA = plan_tri_analysis(F)

# Its Proriol-(α,β,γ) coefficients are:
U = P\(PA*F)

# Similarly, our function's gradient samples are:
Fx = [fx(x[n+1], x[N-n]*w[m+1]) for n in 0:N-1, m in 0:M-1]
Fy = [fy(x[n+1], x[N-n]*w[m+1]) for n in 0:N-1, m in 0:M-1]

# For the partial derivative with respect to x, Olver et al.
# derive simple expressions for the representation of this component
# using a Proriol-(α+1,β,γ+1) series. For the partial derivative with respect
# to y, the analogous formulae result in a Proriol-(α,β+1,γ+1) series.
# These expressions are adapted from https://arxiv.org/abs/1902.04863.
Gx = zeros(Float64, N, M)
for m = 0:M-2
    for n = 0:N-2
        cf1 = m == 0 ? sqrt((n+1)*(n+2m+α+β+γ+3)/(2m+β+γ+2)*(m+γ+1)*8) : sqrt((n+1)*(n+2m+α+β+γ+3)/(2m+β+γ+1)*(m+β+γ+1)/(2m+β+γ+2)*(m+γ+1)*8)
        cf2 = sqrt((n+α+1)*(m+1)/(2m+β+γ+2)*(m+β+1)/(2m+β+γ+3)*(n+2m+β+γ+3)*8)
        Gx[n+1, m+1] = cf1*U[n+2, m+1] + cf2*U[n+1, m+2]
    end
end
Px = plan_tri2cheb(Fx, α+1, β, γ+1)
Ux = Px\(PA*Fx)

Gy = zeros(Float64, N, M)
for m = 0:M-2
    for n = 0:N-2
        Gy[n+1, m+1] = 4*sqrt((m+1)*(m+β+γ+2))*U[n+1, m+2]
    end
end
Py = plan_tri2cheb(Fy, α, β+1, γ+1)
Uy = Py\(PA*Fy)

# The 2-norm relative error in differentiating the Proriol series
# for f(x,y) term-by-term and its sampled gradient is:
hypot(norm(Ux-Gx), norm(Uy-Gy))/hypot(norm(Ux), norm(Uy))

# This error can be improved upon by increasing N and M.

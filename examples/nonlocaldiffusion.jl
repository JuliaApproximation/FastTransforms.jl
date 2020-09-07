# # Nonlocal diffusion on $\mathbb{S}^2$
# This example calculates the spectrum of the nonlocal diffusion operator:
# ```math
# \mathcal{L}_\delta u = \int_{\mathbb{S}^2} \rho_\delta(|\mathbf{x}-\mathbf{y}|)\left[u(\mathbf{x}) - u(\mathbf{y})\right] \,\mathrm{d}\Omega(\mathbf{y}),
# ```
# defined in Eq. (2) of
#
# R. M. Slevinsky, H. Montanelli, and Q. Du, [A spectral method for nonlocal diffusion operators on the sphere](https://doi.org/10.1016/j.jcp.2018.06.024), *J. Comp. Phys.*, **372**:893--911, 2018.
#
# In the above, $0<\delta<2$, $-1<\alpha<1$, and the kernel:
# ```math
# \rho_\delta(|\mathbf{x}-\mathbf{y}|) = \frac{4(1+\alpha)}{\pi \delta^{2+2\alpha}} \frac{\chi_{[0,\delta]}(|\mathbf{x}-\mathbf{y}|)}{|\mathbf{x}-\mathbf{y}|^{2-2\alpha}},
# ```
# where $\chi_I(\cdot)$ is the indicator function on the set $I$.
#
# This nonlocal operator is diagonalized by spherical harmonics:
# ```math
# \mathcal{L}_\delta Y_\ell^m(\mathbf{x}) = \lambda_\ell(\alpha, \delta) Y_\ell^m(\mathbf{x}),
# ```
# and its eigenfunctions are given by the generalized Funk--Hecke formula:
# ```math
# \lambda_\ell(\alpha, \delta) = \frac{(1+\alpha) 2^{2+\alpha}}{\delta^{2+2\alpha}}\int_{1-\delta^2/2}^1 \left[P_\ell(t)-1\right] (1-t)^{\alpha-1} \,\mathrm{d} t.
# ```
# In the paper, the authors use Clenshaw--Curtis quadrature and asymptotic evaluation of Legendre polynomials to achieve $\mathcal{O}(n^2\log n)$ complexity for the evaluation of the first $n$ eigenvalues. With a change of basis, this complexity can be reduced to $\mathcal{O}(n\log n)$.
#
# First, we represent:
# ```math
# P_n(t) - 1 = \sum_{j=0}^{n-1} \left[P_{j+1}(t) - P_j(t)\right] = -\sum_{j=0}^{n-1} (1-t) P_j^{(1,0)}(t).
# ```
# Then, we represent $P_j^{(1,0)}(t)$ with Jacobi polynomials $P_i^{(\alpha,0)}(t)$ and we integrate using [DLMF 18.9.16](https://dlmf.nist.gov/18.9.16):
# ```math
# \int_x^1 P_i^{(\alpha,0)}(t)(1-t)^\alpha\,\mathrm{d}t = \left\{ \begin{array}{cc} \frac{(1-x)^{\alpha+1}}{\alpha+1} & \mathrm{for~}i=0,\\ \frac{1}{2i}(1-x)^{\alpha+1}(1+x)P_{i-1}^{(\alpha+1,1)}(x), & \mathrm{for~}i>0.\end{array}\right.
# ```
# The code below implements this algorithm, making use of the Jacobi--Jacobi transform `plan_jac2jac`.
# For numerical stability, the conversion from Jacobi polynomials $P_j^{(1,0)}(t)$ to $P_i^{(\alpha,0)}(t)$ is divided into conversion from $P_j^{(1,0)}(t)$ to $P_k^{(0,0)}(t)$, before conversion from $P_k^{(0,0)}(t)$ to $P_i^{(\alpha,0)}(t)$.

using FastTransforms, LinearAlgebra

function oprec!(n::Integer, v::AbstractVector, alpha::Real, delta2::Real)
    if n > 0
        v[1] = 1
    end
    if n > 1
        v[2] = (4*alpha+8-(alpha+4)*delta2)/4
    end
    for i = 1:n-2
        v[i+2] = (((2*i+alpha+2)*(2*i+alpha+4)+alpha*(alpha+2))/(2*(i+1)*(2*i+alpha+2))*(2*i+alpha+3)/(i+alpha+3) - delta2/4*(2*i+alpha+3)/(i+1)*(2*i+alpha+4)/(i+alpha+3))*v[i+1] - (i+alpha+1)/(i+alpha+3)*(2*i+alpha+4)/(2*i+alpha+2)*v[i]
    end
    return v
end

function evaluate_lambda(n::Integer, alpha::T, delta::T) where T
    delta2 = delta*delta
    scl = (1+alpha)*(2-delta2/2)

    lambda = Vector{T}(undef, n)

    if n > 0
        lambda[1] = 0
    end
    if n > 1
        lambda[2] = -2
    end

    oprec!(n-2, view(lambda, 3:n), alpha, delta2)

    for i = 2:n-1
        lambda[i+1] *= -scl/(i-1)
    end

    p = plan_jac2jac(T, n-1, zero(T), zero(T), alpha, zero(T))

    lambda[2:end] .= p'lambda[2:end]

    for i = 2:n-1
        lambda[i+1] = ((2i-1)*lambda[i+1] + (i-1)*lambda[i])/i
    end

    for i = 2:n-1
        lambda[i+1] += lambda[i]
    end

    return lambda
end

# The spectrum in `Float64`:
lambda = evaluate_lambda(10, -0.5, 1.0)

# The spectrum in `BigFloat`:
lambdabf = evaluate_lambda(10, parse(BigFloat, "-0.5"), parse(BigFloat, "1.0"))

# The $\infty$-norm relative error:
norm(lambda-lambdabf, Inf)/norm(lambda, Inf)

using FastTransforms

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

# This example calculates the spectrum of the nonlocal diffusion operator:
#
#  \mathcal{L}_\delta u = \int_{\mathbb{S}^2} \rho_\delta(|\mathbf{x}-\mathbf{y}|)\left[u(\mathbf{x}) - u(\mathbf{y})\right] \,\mathrm{d}\Omega(\mathbf{y}),
#
#  defined in Eq. (2) of
#
#    R. M. Slevinsky, H. Montanelli, and Q. Du, [A spectral method for nonlocal diffusion operators on the sphere](https://doi.org/10.1016/j.jcp.2018.06.024), *J. Comp. Phys.*, **372**:893--911, 2018.
#

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

lambda = evaluate_lambda(1024, -0.5, 0.025)
lambdabf = evaluate_lambda(1024, parse(BigFloat, "-0.5"), parse(BigFloat, "0.025"))

norm(lambda64-lambdabf, Inf)/norm(lambda64, Inf)

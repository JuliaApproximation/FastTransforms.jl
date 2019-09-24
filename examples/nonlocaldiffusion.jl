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

"""
This example calculates the spectrum of the nonlocal diffusion operator:

```math
ℒ_δ u = ∫_𝕊² ρ_δ(|𝐱-𝐲|)[u(𝐱) - u(𝐲)] dΩ(𝐲),
```

defined in Eq. (2) of

    R. M. Slevinsky, H. Montanelli, and Q. Du, A spectral method for nonlocal diffusion operators on the sphere, J. Comp. Phys., 372:893--911, 2018.

available at https://doi.org/10.1016/j.jcp.2018.06.024
"""
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

lambda = evaluate_lambda(1024, -0.5, 1.0)
lambdabf = evaluate_lambda(1024, parse(BigFloat, "-0.5"), parse(BigFloat, "1.0"))

norm(lambda-lambdabf, Inf)/norm(lambda, Inf)

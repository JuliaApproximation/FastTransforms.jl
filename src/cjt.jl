function cjt(c::Vector,α,β)
    N = length(c)
    a,b = (α-0.5)%1+0.5,(β-0.5)%1+0.5
    A,B = α-a,β-b

    if α^2 == 0.25 && β^2 == 0.25
        ret = copy(c)
        if α == -0.5 && β == 0.5
            N > 1 && (ret[N] *= 2)
            @inbounds for i=N-1:-1:2 ret[i] = 2(ret[i] - (i-0.5)/2i*ret[i+1]) end
            N > 1 && (ret[1] -= 0.25ret[2])
        elseif α == 0.5 && β == -0.5
            N > 1 && (ret[N] *= 2)
            @inbounds for i=N-1:-1:2 ret[i] = 2(ret[i] + (i-0.5)/2i*ret[i+1]) end
            N > 1 && (ret[1] += 0.25ret[2])
        elseif α == 0.5 && β == 0.5
            N > 1 && (ret[N] *= (4N-2)/N)
            N > 2 && (ret[N-1] *= (4N-6)/(N-1))
            @inbounds for i=N-2:-1:2 ret[i] = (4i-2)/i*(ret[i]+(2i+1)/(8i+8)*ret[i+2]) end
            N > 2 && (ret[1] += 0.1875ret[3])
        end
        for i=1:N ret[i] *= Cx(i-1.0)/sqrtpi end
        return ret
    #elseif α == β
        # Ultraspherical line
    #    ret = jac2cheb(c,α,β,plan_cjt(c,α,β))
    #    return ret
    else
        # General half-open square
        ret = jac2cheb(c,α,β,plan_cjt(c,α,β))
        return ret
    end
end

function icjt(c::Vector,α,β)
    N = length(c)
    a,b = (α-0.5)%1+0.5,(β-0.5)%1+0.5
    A,B = α-a,β-b

    if α^2 == 0.25 && β^2 == 0.25
        ret = copy(c)
        for i=1:N ret[i] *= sqrtpi/Cx(i-1.0) end
        if α == -0.5 && β == 0.5
            N > 1 && (ret[1] += 0.25ret[2])
            @inbounds for i=2:N-1 ret[i] = 0.5ret[i] + (i-0.5)/2i*ret[i+1] end
            N > 1 && (ret[N] = 0.5ret[N])
            return ret
        elseif α == 0.5 && β == -0.5
            N > 1 && (ret[1] -= 0.25ret[2])
            @inbounds for i=2:N-1 ret[i] = 0.5ret[i] - (i-0.5)/2i*ret[i+1] end
            N > 1 && (ret[N] *= 0.5)
            return ret
        elseif α == 0.5 && β == 0.5
            N > 2 && (ret[1] -= 0.1875ret[3])
            @inbounds for i=2:N-2 ret[i] = i/(4i-2)*ret[i] - (2i+1)/(8i+8)*ret[i+2] end
            N > 2 && (ret[N-1] *= (N-1)/(4N-6))
            N > 1 && (ret[N] *= N/(4N-2))
            return ret
        else
            return ret
        end
    #elseif α == β
        # Ultraspherical line
    #    ret = cheb2jac(c,α,β,plan_icjt(c,α,β))
    #    return ret
    else
        # General half-open square
        ret = cheb2jac(c,α,β,plan_icjt(c,α,β))
        return ret
    end
end

cjt(c::Vector,α,β,plan) = jac2cheb(c,α,β,plan)
icjt(c::Vector,α,β,plan) = cheb2jac(c,α,β,plan)
jjt(c::Vector,α,β,γ,δ) = icjt(cjt(c,α,β),γ,δ)

plan_cjt(c::Vector,α,β;M::Int=7) = ForwardChebyshevJacobiPlan(c,α,β,M)
plan_icjt(c::Vector,α,β;M::Int=7) = BackwardChebyshevJacobiPlan(c,α,β,M)

*{T}(p::ChebyshevJacobiPlan{FORWARD,T},c::Vector{T}) = cjt(c,getplanαβ(p)...,p)
*{T}(p::ChebyshevJacobiPlan{BACKWARD,T},c::Vector{T}) = icjt(c,getplanαβ(p)...,p)

"""
    cjt(c,α,β)

Computes the Chebyshev expansion coefficients
given the Jacobi expansion coefficients ``c`` with parameters ``α`` and ``β``.

See also [`icjt`](:func:`icjt`) and [`jjt`](:func:`jjt`).
"""
cjt

"""
    icjt(c,α,β)

Computes the Jacobi expansion coefficients with parameters ``α`` and ``β``
given the Chebyshev expansion coefficients ``c``.

See also [`cjt`](:func:`cjt`) and [`jjt`](:func:`jjt`).
"""
icjt

"""
    jjt(c,α,β,γ,δ)

Computes the Jacobi expansion coefficients with parameters ``γ`` and ``δ``
given the Jacobi expansion coefficients ``c`` with parameters ``α`` and ``β``.

See also [`cjt`](:func:`cjt`) and [`icjt`](:func:`icjt`).
"""
jjt

"""
    plan_cjt(c,α,β;M=7)

Pre-plan optimized DCT-I and DST-I plans and pre-allocate the necessary
arrays, normalization constants, and recurrence coefficients for a forward Chebyshev—Jacobi transform.

``c`` is the vector of coefficients; and,

``α`` and ``β`` are the Jacobi parameters.

Optionally:

``M`` determines the number of terms in Hahn's asymptotic expansion.
"""
plan_cjt

"""
    plan_icjt(c,α,β;M=7)

Pre-plan optimized DCT-I and DST-I plans and pre-allocate the necessary
arrays, normalization constants, and recurrence coefficients for an inverse Chebyshev—Jacobi transform.

``c`` is the vector of coefficients; and,

``α`` and ``β`` are the Jacobi parameters.

Optionally:

``M`` determines the number of terms in Hahn's asymptotic expansion.
"""
plan_icjt

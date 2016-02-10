function cjt(c::Vector,plan)
    α,β = getplanαβ(plan)
    N = length(c)
    if α^2 == 0.25 && β^2 == 0.25
        ret = copy(c)
        if α == -0.5 && β == 0.5
            decrementβ!(ret,α,β)
        elseif α == 0.5 && β == -0.5
            decrementα!(ret,α,β)
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
    #    ret = jac2cheb(c,α,β,plan)
    #    return ret
    else
        # General half-open square
        ret = tosquare!(copy(c),α,β)
        ret = jac2cheb(ret,modαβ(α),modαβ(β),plan)
        return ret
    end
end

function icjt(c::Vector,plan)
    α,β = getplanαβ(plan)
    N = length(c)
    if α^2 == 0.25 && β^2 == 0.25
        ret = copy(c)
        for i=1:N ret[i] *= sqrtpi/Cx(i-1.0) end
        if α == -0.5 && β == 0.5
            incrementβ!(ret,α,β-1)
            return ret
        elseif α == 0.5 && β == -0.5
            incrementα!(ret,α-1,β)
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
    #    ret = cheb2jac(c,α,β,plan)
    #    return ret
    else
        # General half-open square
        ret = cheb2jac(c,modαβ(α),modαβ(β),plan)
        fromsquare!(ret,α,β)
        return ret
    end
end

cjt(c::Vector,α,β) = cjt(c,plan_cjt(c,α,β))
icjt(c::Vector,α,β) = icjt(c,plan_icjt(c,α,β))
jjt(c::Vector,α,β,γ,δ) = icjt(cjt(c,α,β),γ,δ)

function plan_cjt(c::Vector,α,β;M::Int=7)
    P = ForwardChebyshevJacobiPlan(c,modαβ(α),modαβ(β),M)
    P.CJC.α,P.CJC.β = α,β
    P
end
function plan_icjt(c::Vector,α,β;M::Int=7)
    P = BackwardChebyshevJacobiPlan(c,modαβ(α),modαβ(β),M)
    P.CJC.α,P.CJC.β = α,β
    P
end

*{T}(p::ChebyshevJacobiPlan{FORWARD,T},c::Vector{T}) = cjt(c,p)
*{T}(p::ChebyshevJacobiPlan{BACKWARD,T},c::Vector{T}) = icjt(c,p)

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

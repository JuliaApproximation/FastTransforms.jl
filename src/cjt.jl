function cjt(c::AbstractVector,plan)
    α,β = getplanαβ(plan)
    N = length(c)
    N ≤ 1 && return c
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

function icjt(c::AbstractVector,plan)
    α,β = getplanαβ(plan)
    N = length(c)
    N ≤ 1 && return c
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

jjt(c,α,β,γ,δ) = icjt(cjt(c,α,β),γ,δ)

function plan_cjt(c::AbstractVector,α,β;M::Int=7)
    P = ForwardChebyshevJacobiPlan(c,modαβ(α),modαβ(β),M)
    P.CJC.α,P.CJC.β = α,β
    P
end
function plan_icjt(c::AbstractVector,α,β;M::Int=7)
    P = BackwardChebyshevJacobiPlan(c,modαβ(α),modαβ(β),M)
    P.CJC.α,P.CJC.β = α,β
    P
end

for (op,plan_op,D) in ((:cjt,:plan_cjt,:FORWARD),(:icjt,:plan_icjt,:BACKWARD))
    @eval begin
        $op(c,α,β) = $plan_op(c,α,β)*c
        *{T<:AbstractFloat}(p::ChebyshevJacobiPlan{$D,T},c::AbstractVector{T}) = $op(c,p)
        $plan_op{T<:AbstractFloat}(c::AbstractVector{Complex{T}},α,β;M::Int=7) = $plan_op(real(c),α,β;M=M)
        $plan_op(c::AbstractMatrix,α,β;M::Int=7) = $plan_op(slice(c,1:size(c,1)),α,β;M=M)
    end
end

function *{D,T<:AbstractFloat}(p::ChebyshevJacobiPlan{D,T},c::AbstractVector{Complex{T}})
    cr,ci = reim(c)
    complex(p*cr,p*ci)
end

function *(p::ChebyshevJacobiPlan,c::AbstractMatrix)
    m,n = size(c)
    ret = zero(c)
    for j=1:n ret[:,j] = p*slice(c,1:m,j) end
    ret
end

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

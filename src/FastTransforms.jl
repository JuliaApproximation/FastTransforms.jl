__precompile__()
module FastTransforms

using Base

import Base: *

export cjt, icjt, jjt, plan_cjt, plan_icjt

# Other module methods and constants:
#export ChebyshevJacobiPlan, jac2cheb, cheb2jac
#export sqrtpi, pochhammer, stirlingseries, stirlingremainder, Aratio, Cratio, Anαβ
#export Cnmαβ, Cnαβ, Cnmλ, Cnλ, Cx, absf, findmindices!
#export clenshawcurtis, clenshawcurtis_plan, clenshawcurtisweights
#export fejer1, fejer_plan1, fejerweights1
#export fejer2, fejer_plan2, fejerweights2
#export RecurrencePlan, forward_recurrence!, backward_recurrence

include("specialfunctions.jl")
include("clenshawcurtis.jl")
include("fejer.jl")
include("recurrence.jl")

include("ChebyshevJacobiPlan.jl")
include("jac2cheb.jl")
include("cheb2jac.jl")

cjt(c::Vector,α,β) = jac2cheb(c,α,β,plan_cjt(c,α,β))
icjt(c::Vector,α,β) = cheb2jac(c,α,β,plan_icjt(c,α,β))
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

end # module

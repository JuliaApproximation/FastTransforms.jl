plan_fejer1(μ) = FFTW.plan_r2r!(μ, FFTW.REDFT01)

"""
Compute nodes of Fejer's first quadrature rule.
"""
fejernodes1(::Type{T}, N::Int) where T = chebyshevpoints(T, N, Val(1))

"""
Compute weights of Fejer's first quadrature rule with modified Chebyshev moments of the first kind ``\\mu``.
"""
fejerweights1(μ::Vector) = fejerweights1!(copy(μ))
fejerweights1!(μ::Vector) = fejerweights1!(μ, plan_fejer1(μ))
function fejerweights1!(μ::Vector{T}, plan) where T
    N = length(μ)
    rmul!(μ, inv(T(N)))
    return plan*μ
end


plan_fejer2(μ) = FFTW.plan_r2r!(μ, FFTW.RODFT00)

"""
Compute nodes of Fejer's second quadrature rule.
"""
fejernodes2(::Type{T}, N::Int) where T = T[sinpi((N-2k-one(T))/(2N+two(T))) for k=0:N-1]

"""
Compute weights of Fejer's second quadrature rule with modified Chebyshev moments of the second kind ``\\mu``.
"""
fejerweights2(μ::Vector) = fejerweights2!(copy(μ))
fejerweights2!(μ::Vector) = fejerweights2!(μ, plan_fejer2(μ))
function fejerweights2!(μ::Vector{T}, plan) where T
    N = length(μ)
    Np1 = N+one(T)
    rmul!(μ, inv(Np1))
    plan*μ
    @inbounds for i=1:N μ[i] = sinpi(i/Np1)*μ[i] end
    return μ
end

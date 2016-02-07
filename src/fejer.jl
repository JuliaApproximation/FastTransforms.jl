# Compute nodes and weights of the Fejer quadrature rules with a Jacobi weight using modified Chebyshev moments.

fejer_plan1(μ) = FFTW.plan_r2r!(μ, FFTW.REDFT01)
fejer_plan2(μ) = FFTW.plan_r2r!(μ, FFTW.RODFT00)

fejer1{T<:AbstractFloat}(N::Int,α::T,β::T) = fejer1(N,α,β,fejer_plan1(zeros(T,N)))
fejer2{T<:AbstractFloat}(N::Int,α::T,β::T) = fejer2(N,α,β,fejer_plan2(zeros(T,N)))

fejer1{T<:AbstractFloat}(N::Int,α::T,β::T,plan) = T[sinpi((N-2k-one(T))/2N) for k=0:N-1],fejerweights1(N,α,β,plan)
fejer2{T<:AbstractFloat}(N::Int,α::T,β::T,plan) = T[cospi((k+one(T))/(N+one(T))) for k=0:N-1],fejerweights2(N,α,β,plan)


fejerweights1{T<:AbstractFloat}(N::Int,α::T,β::T) = fejerweights1(N,α,β,fejer_plan1(zeros(T,N)))
fejerweights2{T<:AbstractFloat}(N::Int,α::T,β::T) = fejerweights2(N,α,β,fejer_plan2(zeros(T,N)))

function fejerweights1{T<:AbstractFloat}(N::Int,α::T,β::T,plan)
    μ = chebyshevjacobimoments1(N,α,β)
    scale!(μ,inv(T(N)))
    return plan*μ
end

function fejerweights2{T<:AbstractFloat}(N::Int,α::T,β::T,plan)
    μ = chebyshevjacobimoments2(N,α,β)
    Np1 = N+one(T)
    scale!(μ,inv(Np1))
    plan*μ
    @inbounds for i=1:N μ[i] = sinpi(i/Np1)*μ[i] end
    return μ
end

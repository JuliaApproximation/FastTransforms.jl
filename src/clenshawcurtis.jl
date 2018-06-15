clenshawcurtis_plan(μ) = length(μ) > 1 ? FFTW.plan_r2r!(μ, FFTW.REDFT00) : ones(μ)'

"""
Compute nodes and weights of the Clenshaw—Curtis quadrature rule with a Jacobi weight.
"""
clenshawcurtis(N::Int,α::T,β::T) where {T<:AbstractFloat} = clenshawcurtis(N,α,β,clenshawcurtis_plan(zeros(T,N)))
clenshawcurtis(N::Int,α::T,β::T,plan) where {T<:AbstractFloat} = T[cospi(k/(N-one(T))) for k=0:N-1],clenshawcurtisweights(N,α,β,plan)

"""
Compute weights of the Clenshaw—Curtis quadrature rule with a Jacobi weight.
"""
clenshawcurtisweights(N::Int,α::T,β::T) where {T<:AbstractFloat} = clenshawcurtisweights(N,α,β,clenshawcurtis_plan(zeros(T,N)))
function clenshawcurtisweights(N::Int,α::T,β::T,plan) where {T<:AbstractFloat}
    μ = chebyshevjacobimoments1(N,α,β)
    scale!(μ,inv(N-one(T)))
    plan*μ
    μ[1]/=2;μ[N]/=2
    return μ
end

# Chebyshev-T coefficients to values at Clenshaw-Curtis nodes

applyTN_plan(x) = length(x) > 1 ? FFTW.plan_r2r!(x, FFTW.REDFT00) : ones(x)'

applyTN!(x::Vector{T}) where {T<:AbstractFloat} = applyTN!(x,applyTN_plan(x))
function applyTN!(x::Vector{T},plan) where {T<:AbstractFloat}
    x[1] *= 2; x[end] *=2
    plan*x
    scale!(x,half(T))
end
applyTN(x::Vector{T},plan) where {T<:AbstractFloat} = applyTN!(copy(x),plan)
applyTN(x::Vector{T}) where {T<:AbstractFloat} = applyTN!(copy(x))

# Values at Clenshaw-Curtis nodes to Chebyshev-T coefficients

applyTNinv_plan(x) = length(x) > 1 ? FFTW.plan_r2r!(x, FFTW.REDFT00) : ones(x)'

applyTNinv!(x::Vector{T}) where {T<:AbstractFloat} = applyTNinv!(x,applyTNinv_plan(x))
function applyTNinv!(x::Vector{T},plan) where {T<:AbstractFloat}
    plan*x
    x[1] /= 2;x[end] /= 2
    scale!(x,inv(length(x)-one(T)))
end
applyTNinv(x::Vector{T},plan) where {T<:AbstractFloat} = applyTNinv!(copy(x),plan)
applyTNinv(x::Vector{T}) where {T<:AbstractFloat} = applyTNinv!(copy(x))

# sin(nθ) coefficients to values at Clenshaw-Curtis nodes except ±1

applyUN_plan(x) = length(x) > 0 ? FFTW.plan_r2r!(x, FFTW.RODFT00) : ones(x)'

applyUN!(x::AbstractVector{T}) where {T<:AbstractFloat} = applyUN!(x,applyUN_plan(x))
function applyUN!(x::AbstractVector{T},plan) where {T<:AbstractFloat}
    plan*x
    scale!(x,half(T))
end
applyUN(x::AbstractVector{T},plan) where {T<:AbstractFloat} = applyUN!(copy(x),plan)
applyUN(x::AbstractVector{T}) where {T<:AbstractFloat} = applyUN!(copy(x))

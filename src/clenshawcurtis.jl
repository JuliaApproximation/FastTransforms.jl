clenshawcurtis_plan(μ) = length(μ) > 1 ? FFTW.plan_r2r!(μ, FFTW.REDFT00) : ones(μ)'

"""
Compute nodes and weights of the Clenshaw—Curtis quadrature rule with a Jacobi weight.
"""
clenshawcurtis{T<:AbstractFloat}(N::Int,α::T,β::T) = clenshawcurtis(N,α,β,clenshawcurtis_plan(zeros(T,N)))
clenshawcurtis{T<:AbstractFloat}(N::Int,α::T,β::T,plan) = T[cospi(k/(N-one(T))) for k=0:N-1],clenshawcurtisweights(N,α,β,plan)

"""
Compute weights of the Clenshaw—Curtis quadrature rule with a Jacobi weight.
"""
clenshawcurtisweights{T<:AbstractFloat}(N::Int,α::T,β::T) = clenshawcurtisweights(N,α,β,clenshawcurtis_plan(zeros(T,N)))
function clenshawcurtisweights{T<:AbstractFloat}(N::Int,α::T,β::T,plan)
    μ = chebyshevjacobimoments1(N,α,β)
    scale!(μ,inv(N-one(T)))
    plan*μ
    μ[1]/=2;μ[N]/=2
    return μ
end

# Chebyshev-T coefficients to values at Clenshaw-Curtis nodes

applyTN_plan(x) = length(x) > 1 ? FFTW.plan_r2r!(x, FFTW.REDFT00) : ones(x)'

applyTN!{T<:AbstractFloat}(x::Vector{T}) = applyTN!(x,applyTN_plan(x))
function applyTN!{T<:AbstractFloat}(x::Vector{T},plan)
    x[1] *= 2; x[end] *=2
    plan*x
    scale!(x,one(T)/2)
end
applyTN{T<:AbstractFloat}(x::Vector{T},plan) = applyTN!(copy(x),plan)
applyTN{T<:AbstractFloat}(x::Vector{T}) = applyTN!(copy(x))

# Values at Clenshaw-Curtis nodes to Chebyshev-T coefficients

applyTNinv_plan(x) = length(x) > 1 ? FFTW.plan_r2r!(x, FFTW.REDFT00) : ones(x)'

applyTNinv!{T<:AbstractFloat}(x::Vector{T}) = applyTNinv!(x,applyTNinv_plan(x))
function applyTNinv!{T<:AbstractFloat}(x::Vector{T},plan)
    plan*x
    x[1] /= 2;x[end] /= 2
    scale!(x,inv(length(x)-one(T)))
end
applyTNinv{T<:AbstractFloat}(x::Vector{T},plan) = applyTNinv!(copy(x),plan)
applyTNinv{T<:AbstractFloat}(x::Vector{T}) = applyTNinv!(copy(x))

# sin(nθ) coefficients to values at Clenshaw-Curtis nodes

applyUN_plan(x) = length(x) > 2 ? FFTW.plan_r2r!(slice(x,2:length(x)-1), FFTW.RODFT00) : ones(x)'

applyUN!{T<:AbstractFloat}(x::Vector{T}) = applyUN!(x,applyUN_plan(x))
function applyUN!{T<:AbstractFloat}(x::Vector{T},plan)
    x[1] = zero(T); x[end] = zero(T)
    plan*slice(x,2:length(x)-1)
    scale!(x,one(T)/2)
end
applyUN{T<:AbstractFloat}(x::Vector{T},plan) = applyUN!(copy(x),plan)
applyUN{T<:AbstractFloat}(x::Vector{T}) = applyUN!(copy(x))

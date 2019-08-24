plan_clenshawcurtis(μ) = length(μ) > 1 ? FFTW.plan_r2r!(μ, FFTW.REDFT00) : fill!(similar(μ),1)'

"""
Compute nodes of the Clenshaw—Curtis quadrature rule.
"""
clenshawcurtisnodes(::Type{T}, N::Int) where T = chebyshevpoints(T, N; kind = 2)

"""
Compute weights of the Clenshaw—Curtis quadrature rule with modified Chebyshev moments of the first kind ``\\mu``.
"""
clenshawcurtisweights(μ::Vector) = clenshawcurtisweights!(copy(μ))
clenshawcurtisweights!(μ::Vector) = clenshawcurtisweights!(μ, plan_clenshawcurtis(μ))
function clenshawcurtisweights!(μ::Vector{T}, plan) where T
    N = length(μ)
    rmul!(μ, inv(N-one(T)))
    plan*μ
    μ[1] *= half(T); μ[N] *= half(T)
    return μ
end



# Chebyshev-T coefficients to values at Clenshaw-Curtis nodes

applyTN_plan(x) = length(x) > 1 ? FFTW.plan_r2r!(x, FFTW.REDFT00) : fill!(similar(x),1)'

applyTN!(x::Vector{T}) where {T<:AbstractFloat} = applyTN!(x,applyTN_plan(x))

function applyTN!(x::Vector{T},plan) where T<:AbstractFloat
    x[1] *= 2; x[end] *=2
    plan*x
    rmul!(x,half(T))
end
applyTN(x::Vector{T},plan) where {T<:AbstractFloat} = applyTN!(copy(x),plan)
applyTN(x::Vector{T}) where {T<:AbstractFloat} = applyTN!(copy(x))

# Values at Clenshaw-Curtis nodes to Chebyshev-T coefficients

applyTNinv_plan(x) = length(x) > 1 ? FFTW.plan_r2r!(x, FFTW.REDFT00) : fill!(similar(x),1)'

applyTNinv!(x::Vector{T}) where {T<:AbstractFloat} = applyTNinv!(x,applyTNinv_plan(x))

function applyTNinv!(x::Vector{T},plan) where T<:AbstractFloat
    plan*x
    x[1] /= 2;x[end] /= 2
    rmul!(x,inv(length(x)-one(T)))
end
applyTNinv(x::Vector{T},plan) where {T<:AbstractFloat} = applyTNinv!(copy(x),plan)
applyTNinv(x::Vector{T}) where {T<:AbstractFloat} = applyTNinv!(copy(x))

# sin(nθ) coefficients to values at Clenshaw-Curtis nodes except ±1

applyUN_plan(x) = length(x) > 0 ? FFTW.plan_r2r!(x, FFTW.RODFT00) : fill!(similar(x),1)'

applyUN!(x::AbstractVector{T}) where {T<:AbstractFloat} = applyUN!(x,applyUN_plan(x))

function applyUN!(x::AbstractVector{T},plan) where T<:AbstractFloat
    plan*x
    rmul!(x,half(T))
end
applyUN(x::AbstractVector{T},plan) where {T<:AbstractFloat} = applyUN!(copy(x),plan)
applyUN(x::AbstractVector{T}) where {T<:AbstractFloat} = applyUN!(copy(x))

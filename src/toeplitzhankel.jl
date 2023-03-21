"""
Store a diagonally-scaled Toeplitz∘Hankel matrix:
    DL(T∘H)DR
where the Hankel matrix `H` is non-negative definite. This allows a Cholesky decomposition in 𝒪(K²N) operations and 𝒪(KN) storage, K = log N log ɛ⁻¹.
"""
struct ToeplitzHankelPlan{S, N, M, TP<:ToeplitzPlan{S,N}} <: Plan{S}
    T::TP
    C::NTuple{M,Vector{Vector{S}}}
    DL::NTuple{M,Vector{S}}
    DR::NTuple{M,Vector{S}}
    tmp1::Array{S,N}
    tmp2::Array{S,N}
    dims::NTuple{M,Int}
    function ToeplitzHankelPlan{S,N,M,TP}(T::TP, C, DL, DR, dims) where {S,TP,N,M}
        tmp1 = Array{S}(undef, map(length,DL))
        new{S,N,M,TP}(T, C, DL, DR, tmp1, similar(tmp1))
    end
end

function ToeplitzHankelPlan(T::ToeplitzPlan, C::Vector, DL::AbstractVector, DR::AbstractVector)
    S = promote_type(eltype(T), eltype(C[1]), eltype(DL), eltype(DR))
    ToeplitzHankelPlan{S, 1, 1, typeof(T)}(T, (C,), (convert(Vector{S},DL),), (convert(Vector{S},DR),), 1)
end
ToeplitzHankelPlan(T::ToeplitzPlan, C::Matrix) =
    ToeplitzHankelPlan(T, C, ones(size(T, 1)),ones(size(T,2)))

function *(P::ToeplitzHankelPlan, v::AbstractVector)
    v .= P.DR[1] .* v
    toeplitzcholmult!(P.T, P.C[1], v, P.tmp1, P.tmp2)
    v .= P.DL[1] .* v
end

function hankel_partialchol(v::Vector{T}) where T
    # Assumes positive definite
    σ = T[]
    n = (length(v)+2) ÷ 2
    C = Vector{T}[]
    d = v[1:2:end] # diag of H
    @assert length(v) ≥ 2n-1
    reltol = maximum(abs,d)*eps(T)*log(n)
    for k = 1:n
        mx,idx = findmax(d)
        if mx ≤ reltol break end
        push!(σ, inv(mx))
        push!(C, v[idx:n+idx-1])
        for j = 1:k-1
            nCjidxσj = -C[j][idx]*σ[j]
            LinearAlgebra.axpy!(nCjidxσj, C[j], C[k])
        end
        @inbounds for p=1:n
            d[p] -= C[k][p]^2/mx
        end
    end
    for k=1:length(σ) rmul!(C[k],sqrt(σ[k])) end
    C
end

function hankel_partialchol(v::AbstractVector, D::AbstractVector)
    # Assumes positive definite
    T = promote_type(eltype(v),eltype(D))
    σ = T[]
    n = (length(v)+2) ÷ 2
    C = Vector{T}[]
    d = v[1:2:end].*D.^2
    @assert length(v) ≥ 2n-1
    reltol = maximum(abs,d)*eps(T)*log(n)
    for k = 1:n
        mx,idx = findmax(d)
        if mx ≤ reltol break end
        push!(σ,inv(mx))
        push!(C,v[idx:n+idx-1] .* D .* D[idx])
        for j = 1:k-1
            nCjidxσj = -C[j][idx]*σ[j]
            LinearAlgebra.axpy!(nCjidxσj, C[j], C[k])
        end
        @simd for p=1:n
            @inbounds d[p]-=C[k][p]^2/mx
        end
    end
    for k = 1:length(σ) rmul!(C[k],sqrt(σ[k])) end
    C
end

function toeplitzcholmult!(T, C, v, tmp, ret)
    K = length(C)
    fill!(ret, zero(eltype(ret)))
    for k = K:-1:1
        tmp .= C[k] .* v
        T * tmp
        tmp .= C[k] .* tmp
        ret .= ret .+ tmp
    end
    copyto!(v, ret)
end


# Diagonally-scaled Toeplitz∘Hankel polynomial transforms

function _leg2chebTH_λt(::Type{S}, n) where S
    λ = Λ.(0:half(S):n-1)
    t = zeros(S,n)
    t[1:2:end] .= 2 .* view(λ, 1:2:n) ./ π
    λ,t
end

function leg2chebTH(::Type{S}, (n,)) where S
    λ,t = _leg2chebTH_λt(S, n)
    T = plan_uppertoeplitz!(t)
    DL = ones(S,n)
    DL[1] /= 2
    T, hankel_partialchol(λ), DL
end

function cheb2legTH(::Type{S}, (n,)) where S
    t = zeros(S,n-1)
    t[1:2:end] = Λ.(0:one(S):div(n-2,2), -half(S), one(S))
    T = plan_uppertoeplitz!(t)
    h = Λ.(1:half(S):n-1, zero(S), 3half(S))
    D = 1:one(S):n-1
    DL = (3half(S):n-half(S))./D
    DR = -(one(S):n-one(S))./4D
    T, hankel_partialchol(h,D), DL,DR
end

function leg2chebuTH(::Type{S}, (n,)) where S
    λ = Λ.(0:half(S):n-1)
    t = zeros(S,n)
    t[1:2:end] = λ[1:2:n]./(((1:2:n).-2))
    T = plan_uppertoeplitz!(-2t/π)
    h = λ./((1:2n-1).+1)
    T,hankel_partialchol(h)
end

function ultra2ultraTH(::Type{S}, (n,), λ₁, λ₂) where S
    @assert abs(λ₁-λ₂) < 1
    DL = (zero(S):n-one(S)) .+ λ₂
    jk = 0:half(S):n-1
    t = zeros(S,n)
    t[1:2:n] = Λ.(jk,λ₁-λ₂,one(S))[1:2:n]
    T = plan_uppertoeplitz!(lmul!(inv(gamma(λ₁-λ₂)),t))
    h = Λ.(jk,λ₁,λ₂+one(S))
    lmul!(gamma(λ₂)/gamma(λ₁),h)
    DR = ones(S,n)
    T,hankel_partialchol(h),DL,DR
end

function jac2jacTH(::Type{S}, (n,), α, β, γ, δ) where S
    if β == δ
        @assert abs(α-γ) < 1
        @assert α+β > -1
        jk = 0:n-1
        DL = (2jk .+ γ .+ β .+ 1).*Λ.(jk,γ+β+1,β+1)
        T = plan_uppertoeplitz!(Λ.(jk,α-γ,1))
        h = Λ.(0:2n-2,α+β+1,γ+β+2)
        DR = Λ.(jk,β+1,α+β+1)./gamma(α-γ)
        T,hankel_partialchol(h),DL,DR
    elseif α == γ
        T,H,DL,DR = jac2jacTH(S,n,β,α,δ,γ)
        ve = T.ve
        @inbounds for k = 2:2:length(ve)
            ve[k] *= -1
        end
        plan_uppertoeplitz!(ve),H,DL,DR
    else
        throw(ArgumentError("Cannot create Toeplitz dot Hankel, use a sequence of plans."))
    end
end

struct ChebyshevToLegendrePlanTH{TH}
    toeplitzhankel::TH
end

ChebyshevToLegendrePlanTH(::Type{S},n) where {S} = ChebyshevToLegendrePlanTH(plan_th_cheb2leg(S,n))

function *(P::ChebyshevToLegendrePlanTH, v::AbstractVector)
    w = zero(v)
    S,n = eltype(v),length(v)
    w[1:2:end] = -one(S)./(one(S):two(S):n)./(-one(S):two(S):n-two(S))
    [dot(w,v); P.toeplitzhankel*view(v,2:n)]
end

plan_th_leg2cheb!(::Type{S}, n) where {S} = ToeplitzHankelPlan(leg2chebTH(S, n)..., ones(S, n))
plan_th_cheb2leg!(::Type{S}, n) where {S} = ChebyshevToLegendrePlanTH(ToeplitzHankelPlan(cheb2legTH(S, n)...))
plan_th_leg2chebu!(::Type{S}, (n,)) where {S} = ToeplitzHankelPlan(leg2chebuTH(S, (n,))..., 1:n, ones(S, n))
plan_th_ultra2ultra!(::Type{S}, n, λ₁, λ₂) where {S} = ToeplitzHankelPlan(ultra2ultraTH(S, n, λ₁, λ₂)...)
plan_th_jac2jac!(::Type{S},n, α, β, γ, δ) where {S} = ToeplitzHankelPlan(jac2jacTH(S, n, α, β, γ, δ)...)


th_leg2cheb(v, dims...) = plan_th_leg2cheb!(eltype(v), size(v), dims...)*copy(v)
th_cheb2leg(v, dims...) = plan_th_cheb2leg!(eltype(v), size(v), dims...)*copy(v)
th_leg2chebu(v, dims...) = plan_th_leg2chebu!(eltype(v), size(v), dims...)*copy(v)
th_ultra2ultra(v, λ₁, λ₂, dims...) = plan_th_ultra2ultra!(eltype(v),size(v),λ₁,λ₂, dims...)*copy(v)
th_jac2jac(v, α, β, γ, δ, dims...) = plan_th_jac2jac!(eltype(v),size(v),α,β,γ,δ, dims...)*copy(v)
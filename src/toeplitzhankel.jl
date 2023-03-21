"""
Store a diagonally-scaled Toeplitz∘Hankel matrix:
    DL(T∘H)DR
where the Hankel matrix `H` is non-negative definite. This allows a Cholesky decomposition in 𝒪(K²N) operations and 𝒪(KN) storage, K = log N log ɛ⁻¹.
"""
struct ToeplitzHankelPlan{S, N, M, N1, TP<:ToeplitzPlan{S,N1}} <: Plan{S}
    T::TP
    L::NTuple{M,Matrix{S}}
    R::NTuple{M,Matrix{S}}
    tmp::Array{S,N1}
    dims::NTuple{M,Int}
    function ToeplitzHankelPlan{S,N,M,N1,TP}(T::TP, L, R, dims) where {S,TP,N,N1,M}
        tmp = Array{S}(undef, size(T))
        new{S,N,M,N1,TP}(T, L, R, tmp, dims)
    end
    ToeplitzHankelPlan{S,N,M,N1,TP}(T::TP, L, R, dims::Int) where {S,TP,N,N1,M} = 
        ToeplitzHankelPlan{S,N,M,N1,TP}(T, L, R, (dims,))
end

ToeplitzHankelPlan(T::ToeplitzPlan{S,2}, L::Matrix, R::Matrix) where S =
    ToeplitzHankelPlan{S, 1, 1, 2, typeof(T)}(T, (L,), (R,), 1)

ToeplitzHankelPlan(T::ToeplitzPlan{S,3}, L::Matrix, R::Matrix, dims) where S =
    ToeplitzHankelPlan{S, 2, 1,3, typeof(T)}(T, (L,), (R,), dims)    


function *(P::ToeplitzHankelPlan{<:Any,1}, v::AbstractVector)
    (R,),(L,),tmp = P.R,P.L,P.tmp
    tmp .= R .* v
    P.T * tmp
    tmp .= L .* tmp
    sum!(v, tmp)
end

function *(P::ToeplitzHankelPlan{<:Any,2,1}, v::AbstractMatrix)
    (R,),(L,),tmp = P.R,P.L,P.tmp
    if P.dims == (1,)
        tmp .=  reshape(R,size(R,1),1,size(R,2)) .* v
        P.T * tmp
        tmp .=  reshape(L,size(L,1),1,size(L,2)) .* tmp
        sum!(v, tmp)
    else
        tmp .=  reshape(R,1,size(R,1),size(R,2)) .* v
        P.T * tmp
        tmp .=  reshape(L,1,size(L,1),size(L,2)) .* tmp
        sum!(v, tmp)
    end
    v
end

_cholmul!(tmp, (C,)::Tuple{Any}, k, v, ::Val{1}) = broadcast!(*, tmp, view(C,:,k), v)
_cholmul!(tmp, (C,)::Tuple{Any}, k, v, ::Val{2}) = broadcast!(*, tmp, v, transpose(view(C,:,k)))

function toeplitzcholmult!(T, C, v, tmp, ret, dims)
    K = size(C[1],2)
    fill!(ret, zero(eltype(ret)))
    for k = K:-1:1
        _cholmul!(tmp, C, k, v, dims)
        T * tmp
        _cholmul!(tmp, C, k, tmp, dims)
        ret .= ret .+ tmp
    end
    copyto!(v, ret)
end


# partial cholesky for a Hankel matrix

function hankel_partialchol(v::Vector{T}) where T
    # Assumes positive definite
    σ = T[]
    n = (length(v)+2) ÷ 2
    C = Matrix{T}(undef, n, n)
    d = v[1:2:end] # diag of H
    @assert length(v) ≥ 2n-1
    reltol = maximum(abs,d)*eps(T)*log(n)
    r = 0
    for k = 1:n
        mx,idx = findmax(d)
        if mx ≤ reltol break end
        push!(σ, inv(mx))
        C[:,k] .= view(v,idx:n+idx-1)
        for j = 1:k-1
            nCjidxσj = -C[idx,j]*σ[j]
            LinearAlgebra.axpy!(nCjidxσj, view(C,:,j), view(C,:,k))
        end
        @inbounds for p=1:n
            d[p] -= C[p,k]^2/mx
        end
        r += 1
    end
    for k=1:length(σ) rmul!(view(C,:,k), sqrt(σ[k])) end
    C[:,1:r]
end

function hankel_partialchol(v::AbstractVector, D::AbstractVector)
    # Assumes positive definite
    T = promote_type(eltype(v),eltype(D))
    σ = T[]
    n = (length(v)+2) ÷ 2
    C = Matrix{T}(undef, n, n)
    d = v[1:2:end].*D.^2
    @assert length(v) ≥ 2n-1
    reltol = maximum(abs,d)*eps(T)*log(n)
    for k = 1:n
        mx,idx = findmax(d)
        if mx ≤ reltol break end
        push!(σ,inv(mx))
        C[:,k] .= v[idx:n+idx-1] .* D .* D[idx]
        for j = 1:k-1
            nCjidxσj = -C[idx,j]*σ[j]
            LinearAlgebra.axpy!(nCjidxσj, view(C,:,j), view(C,:,k))
        end
        @simd for p=1:n
            @inbounds d[p]-=C[p,k]^2/mx
        end
    end
    for k = 1:length(σ) rmul!(view(C,:,k),sqrt(σ[k])) end
    C
end


# Diagonally-scaled Toeplitz∘Hankel polynomial transforms



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

function _leg2chebTH_λt(::Type{S}, n) where S
    λ = Λ.(0:half(S):n-1)
    t = zeros(S,n)
    t[1:2:end] .= 2 .* view(λ, 1:2:n) ./ π
    λ,t
end


function plan_th_leg2cheb!(::Type{S}, (n,)::Tuple{Int}, dims...) where {S}
    λ,t = _leg2chebTH_λt(S, n)
    C = hankel_partialchol(λ)
    T = plan_uppertoeplitz!(t, (length(t), size(C,2)), 1)
    DL = ones(S,n)
    DL[1] /= 2
    ToeplitzHankelPlan(T, DL .* C, C)
end

function plan_th_leg2cheb!(::Type{S}, (m,n)::NTuple{2,Int}, dims::Int) where {S}
    if dims == 1
        λ,t = _leg2chebTH_λt(S, m)
        C = hankel_partialchol(λ)
        T = plan_uppertoeplitz!(t, (m,n,size(C,2)), dims)
        DL = ones(S,m)
        DL[1] /= 2
        ToeplitzHankelPlan(T, DL .* C, C, dims)
    else
        @assert dims == 2
        λ,t = _leg2chebTH_λt(S, n)
        C = hankel_partialchol(λ)
        T = plan_uppertoeplitz!(t, (m,n), dims)
        DL = ones(S,n)
        DL[1] /= 2
        ToeplitzHankelPlan(T, hankel_partialchol(λ), DL, ones(S, n), dims)
    end
end

function plan_th_leg2cheb!(::Type{S}, (m,n)::NTuple{2,Int}, dims::NTuple{2,Int}) where {S} 
    @assert dims == (1,2)
    λ1,t1 = _leg2chebTH_λt(S, m)
    λ2,t2 = _leg2chebTH_λt(S, n)
    T = plan_uppertoeplitz!((t1,t2), (m,n), dims)
    DL1 = ones(S,m); DL1[1] /= 2
    DL2 = ones(S,n); DL2[1] /= 2
    ToeplitzHankelPlan(T, hankel_partialchol(λ), (DL1,DL2), (ones(S, m),ones(S, n)), dims)
end

plan_th_leg2cheb!(::Type{S}, (m,n)::NTuple{2,Int}) where {S} = plan_th_leg2cheb!(S, (m,n), (1,2))
plan_th_leg2cheb!(arr::AbstractArray{T}, dims...) where T = plan_th_leg2cheb!(T, size(arr), dims...)


function plan_th_cheb2leg!(::Type{S}, (n,)::Tuple{Int}) where {S}
    t = zeros(S,n-1)
    t[1:2:end] = Λ.(0:one(S):div(n-2,2), -half(S), one(S))
    h = Λ.(1:half(S):n-1, zero(S), 3half(S))
    D = 1:one(S):n-1
    DL = (3half(S):n-half(S))./D
    DR = -(one(S):n-one(S))./4D
    C = hankel_partialchol(h,D)
    T = plan_uppertoeplitz!(t, (length(t), size(C,2)), 1)
    ChebyshevToLegendrePlanTH(ToeplitzHankelPlan(T, DL .* C, DR .* C))
end
function plan_th_leg2chebu!(::Type{S}, (n,)) where {S}
    λ = Λ.(0:half(S):n-1)
    t = zeros(S,n)
    t[1:2:end] = λ[1:2:n]./(((1:2:n).-2))
    h = λ./((1:2n-1).+1)
    C = hankel_partialchol(h)
    T = plan_uppertoeplitz!(-2t/π, (length(t), size(C,2)), 1) 
    ToeplitzHankelPlan(T, (1:n) .* C, C)
end
function plan_th_ultra2ultra!(::Type{S}, (n,)::Tuple{Int}, λ₁, λ₂) where {S}
    @assert abs(λ₁-λ₂) < 1
    DL = (zero(S):n-one(S)) .+ λ₂
    jk = 0:half(S):n-1
    t = zeros(S,n)
    t[1:2:n] = Λ.(jk,λ₁-λ₂,one(S))[1:2:n]
    h = Λ.(jk,λ₁,λ₂+one(S))
    lmul!(gamma(λ₂)/gamma(λ₁),h)
    C = hankel_partialchol(h)
    T = plan_uppertoeplitz!(lmul!(inv(gamma(λ₁-λ₂)),t), (length(t), size(C,2)), 1) 
    ToeplitzHankelPlan(T, DL .* C, C)
end

function plan_th_jac2jac!(::Type{S}, (n,), α, β, γ, δ) where {S}
    if β == δ
        @assert abs(α-γ) < 1
        @assert α+β > -1
        jk = 0:n-1
        DL = (2jk .+ γ .+ β .+ 1).*Λ.(jk,γ+β+1,β+1)
        t = Λ.(jk,α-γ,1)
        h = Λ.(0:2n-2,α+β+1,γ+β+2)
        DR = Λ.(jk,β+1,α+β+1)./gamma(α-γ)
        C = hankel_partialchol(h)
        T = plan_uppertoeplitz!(t, (length(t), size(C,2)), 1) 
    elseif α == γ
        jk = 0:n-1
        DL = (2jk .+ δ .+ α .+ 1).*Λ.(jk,δ+α+1,α+1)
        h = Λ.(0:2n-2,α+β+1,δ+α+2)
        DR = Λ.(jk,α+1,α+β+1)./gamma(β-δ)
        t = alternatesign!(Λ.(jk,β-δ,1))
        C = hankel_partialchol(h)
        T = plan_uppertoeplitz!(t, (length(t), size(C,2)), 1) 
    else
        throw(ArgumentError("Cannot create Toeplitz dot Hankel, use a sequence of plans."))
    end
    
    ToeplitzHankelPlan(T, DL .* C, DR .* C)
end
function alternatesign!(v)
    @inbounds for k = 2:2:length(v)
        v[k] = -v[k]
    end
    v
end


th_leg2cheb(v, dims...) = plan_th_leg2cheb!(eltype(v), size(v), dims...)*copy(v)
th_cheb2leg(v, dims...) = plan_th_cheb2leg!(eltype(v), size(v), dims...)*copy(v)
th_leg2chebu(v, dims...) = plan_th_leg2chebu!(eltype(v), size(v), dims...)*copy(v)
th_ultra2ultra(v, λ₁, λ₂, dims...) = plan_th_ultra2ultra!(eltype(v),size(v),λ₁,λ₂, dims...)*copy(v)
th_jac2jac(v, α, β, γ, δ, dims...) = plan_th_jac2jac!(eltype(v),size(v),α,β,γ,δ, dims...)*copy(v)
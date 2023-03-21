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
        tmp1 = Array{S}(undef, size(T))
        new{S,N,M,TP}(T, C, DL, DR, tmp1, similar(tmp1), dims)
    end
    ToeplitzHankelPlan{S,N,M,TP}(T::TP, C, DL, DR, dims::Int) where {S,TP,N,M} = 
        ToeplitzHankelPlan{S,N,M,TP}(T, C, DL, DR, (dims,))
end

ToeplitzHankelPlan(T::ToeplitzPlan{S,1}, C::Vector, DL::AbstractVector, DR::AbstractVector) where S =
    ToeplitzHankelPlan{S, 1, 1, typeof(T)}(T, (C,), (convert(Vector{S},DL),), (convert(Vector{S},DR),), 1)

ToeplitzHankelPlan(T::ToeplitzPlan{S,2}, C::Vector, DL::AbstractVector, DR::AbstractVector, dims) where S =
    ToeplitzHankelPlan{S, 2, 1, typeof(T)}(T, (C,), (convert(Vector{S},DL),), (convert(Vector{S},DR),), dims)    

ToeplitzHankelPlan(T::ToeplitzPlan, C::Matrix) =
    ToeplitzHankelPlan(T, C, ones(size(T, 1)),ones(size(T,2)))

function *(P::ToeplitzHankelPlan{<:Any,1}, v::AbstractVector)
    DR, = P.DR
    DL, = P.DL
    v .= DR .* v
    toeplitzcholmult!(P.T, P.C, v, P.tmp1, P.tmp2, Val(1))
    v .= DL .* v
end

function *(P::ToeplitzHankelPlan{<:Any,2,1}, v::AbstractMatrix)
    DR, = P.DR
    DL, = P.DL
    if P.dims == (1,)
        v .= DR .* v
        toeplitzcholmult!(P.T, P.C, v, P.tmp1, P.tmp2, Val(1))
        v .= DL .* v
    else
        v .= v .* transpose(DR)
        toeplitzcholmult!(P.T, P.C, v, P.tmp1, P.tmp2, Val(2))
        v .= v .* transpose(DL)
    end
end



_cholmul!(tmp, (C,)::Tuple{Any}, k, v, ::Val{1}) = broadcast!(*, tmp, C[k], v)
_cholmul!(tmp, (C,)::Tuple{Any}, k, v, ::Val{2}) = broadcast!(*, tmp, v, transpose(C[k]))

function toeplitzcholmult!(T, C, v, tmp, ret, dims)
    K = length(C[1])
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


# Diagonally-scaled Toeplitz∘Hankel polynomial transforms



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

function alternatesign!(v)
    @inbounds for k = 2:2:length(v)
        v[k] = -v[k]
    end
    v
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
    elseif α == γ
        jk = 0:n-1
        DL = (2jk .+ δ .+ α .+ 1).*Λ.(jk,δ+α+1,α+1)
        T = plan_uppertoeplitz!(alternatesign!(Λ.(jk,β-δ,1)))
        h = Λ.(0:2n-2,α+β+1,δ+α+2)
        DR = Λ.(jk,α+1,α+β+1)./gamma(β-δ)
    else
        throw(ArgumentError("Cannot create Toeplitz dot Hankel, use a sequence of plans."))
    end
    T,hankel_partialchol(h),DL,DR
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

function _leg2chebTH_λt(::Type{S}, n) where S
    λ = Λ.(0:half(S):n-1)
    t = zeros(S,n)
    t[1:2:end] .= 2 .* view(λ, 1:2:n) ./ π
    λ,t
end

function leg2chebTH(::Type{S}, (n,)) where S

    
end


function plan_th_leg2cheb!(::Type{S}, (n,)::Tuple{Int}, dims...) where {S}
    λ,t = _leg2chebTH_λt(S, n)
    T = plan_uppertoeplitz!(t)
    DL = ones(S,n)
    DL[1] /= 2
    ToeplitzHankelPlan(T, hankel_partialchol(λ), DL, ones(S, n))
end

function plan_th_leg2cheb!(::Type{S}, (m,n)::NTuple{2,Int}, dims::Tuple{Int}) where {S}
    if tuple(dims...) == (1,)
        λ,t = _leg2chebTH_λt(S, m)
        T = plan_uppertoeplitz!(t, (m,n), dims)
        DL = ones(S,m)
        DL[1] /= 2
        ToeplitzHankelPlan(T, hankel_partialchol(λ), DL, ones(S, m), dims)
    else
        @assert tuple(dims...) == (2,)
        λ,t = _leg2chebTH_λt(S, n)
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


plan_th_cheb2leg!(::Type{S}, n) where {S} = ChebyshevToLegendrePlanTH(ToeplitzHankelPlan(cheb2legTH(S, n)...))
plan_th_leg2chebu!(::Type{S}, (n,)) where {S} = ToeplitzHankelPlan(leg2chebuTH(S, (n,))..., 1:n, ones(S, n))
plan_th_ultra2ultra!(::Type{S}, n, λ₁, λ₂) where {S} = ToeplitzHankelPlan(ultra2ultraTH(S, n, λ₁, λ₂)...)
plan_th_jac2jac!(::Type{S},n, α, β, γ, δ) where {S} = ToeplitzHankelPlan(jac2jacTH(S, n, α, β, γ, δ)...)


th_leg2cheb(v, dims...) = plan_th_leg2cheb!(eltype(v), size(v), dims...)*copy(v)
th_cheb2leg(v, dims...) = plan_th_cheb2leg!(eltype(v), size(v), dims...)*copy(v)
th_leg2chebu(v, dims...) = plan_th_leg2chebu!(eltype(v), size(v), dims...)*copy(v)
th_ultra2ultra(v, λ₁, λ₂, dims...) = plan_th_ultra2ultra!(eltype(v),size(v),λ₁,λ₂, dims...)*copy(v)
th_jac2jac(v, α, β, γ, δ, dims...) = plan_th_jac2jac!(eltype(v),size(v),α,β,γ,δ, dims...)*copy(v)
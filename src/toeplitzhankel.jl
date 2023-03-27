"""
Store a diagonally-scaled Toeplitz∘Hankel matrix:
    DL(T∘H)DR
where the Hankel matrix `H` is non-negative definite. This allows a Cholesky decomposition in 𝒪(K²N) operations and 𝒪(KN) storage, K = log N log ɛ⁻¹.
"""
struct ToeplitzHankelPlan{S, N, M, N1, TP<:ToeplitzPlan{S,N1}} <: Plan{S}
    T::NTuple{M,TP}
    L::NTuple{M,Matrix{S}}
    R::NTuple{M,Matrix{S}}
    tmp::Array{S,N1}
    dims::NTuple{M,Int}
    function ToeplitzHankelPlan{S,N,M,N1,TP}(T::NTuple{M,TP}, L, R, dims) where {S,TP,N,N1,M}
        tmp = Array{S}(undef, max.(size.(T)...)...)
        new{S,N,M,N1,TP}(T, L, R, tmp, dims)
    end
    ToeplitzHankelPlan{S,N,M,N1,TP}(T::NTuple{M,TP}, L, R, dims::Int) where {S,TP,N,N1,M} =
        ToeplitzHankelPlan{S,N,M,N1,TP}(T, L, R, (dims,))
end

ToeplitzHankelPlan(T::ToeplitzPlan{S,2}, L::Matrix, R::Matrix, dims=1) where S =
    ToeplitzHankelPlan{S, 1, 1, 2, typeof(T)}((T,), (L,), (R,), dims)

ToeplitzHankelPlan(T::ToeplitzPlan{S,3}, L::Matrix, R::Matrix, dims) where S =
    ToeplitzHankelPlan{S, 2, 1,3, typeof(T)}((T,), (L,), (R,), dims)

ToeplitzHankelPlan(T::NTuple{2,TP}, L::Tuple, R::Tuple, dims) where {S,TP<:ToeplitzPlan{S,3}} =
    ToeplitzHankelPlan{S, 2,2,3, TP}(T, L, R, dims)


function *(P::ToeplitzHankelPlan{<:Any,1}, v::AbstractVector)
    (R,),(L,),(T,),tmp = P.R,P.L,P.T,P.tmp
    tmp .= R .* v
    T * tmp
    tmp .= L .* tmp
    sum!(v, tmp)
end

function _th_applymul1!(v, T, L, R, tmp)
    N = size(R,2)
    m,n = size(v)
    tmp[1:m,1:n,1:N] .=  reshape(R,size(R,1),1,N) .* v
    T * view(tmp,1:m,1:n,1:N)
    view(tmp,1:m,1:n,1:N) .*=  reshape(L,size(L,1),1,N)
    sum!(v, view(tmp,1:m,1:n,1:N))
end

function _th_applymul2!(v, T, L, R, tmp)
    N = size(R,2)
    m,n = size(v)
    tmp[1:m,1:n,1:N] .=  reshape(R,1,size(R,1),N) .* v
    T * view(tmp,1:m,1:n,1:N)
    view(tmp,1:m,1:n,1:N) .*=  reshape(L,1,size(L,1),N)
    sum!(v, view(tmp,1:m,1:n,1:N))
end


function *(P::ToeplitzHankelPlan{<:Any,2,1}, v::AbstractMatrix)
    (R,),(L,),(T,),tmp = P.R,P.L,P.T,P.tmp
    if P.dims == (1,)
        _th_applymul1!(v, T, L, R, tmp)
    else
        _th_applymul2!(v, T, L, R, tmp)
    end
    v
end

function *(P::ToeplitzHankelPlan{<:Any,2,2}, v::AbstractMatrix)
    (R1,R2),(L1,L2),(T1,T2),tmp = P.R,P.L,P.T,P.tmp

    _th_applymul1!(v, T1, L1, R1, tmp)
    _th_applymul2!(v, T2, L2, R2, tmp)

    v
end

# partial cholesky for a Hankel matrix

function hankel_partialchol(v::Vector{T}) where T
    # Assumes positive definite
    σ = T[]
    n = isempty(v) ? 0 : (length(v)+2) ÷ 2
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

# cholesky for D .* H .* D'
function hankel_partialchol(v::Vector, D::AbstractVector)
    T = promote_type(eltype(v), eltype(D))
    # Assumes positive definite
    σ = T[]
    n = isempty(v) ? 0 : (length(v)+2) ÷ 2
    C = Matrix{T}(undef, n, n)
    d = v[1:2:end] .* D.^2 # diag of D .* H .* D'
    @assert length(v) ≥ 2n-1
    reltol = maximum(abs,d)*eps(T)*log(n)
    r = 0
    for k = 1:n
        mx,idx = findmax(d)
        if mx ≤ reltol break end
        push!(σ, inv(mx))
        C[:,k] .= view(v,idx:n+idx-1) .*D.*D[idx]
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



# Diagonally-scaled Toeplitz∘Hankel polynomial transforms



struct ChebyshevToLegendrePlanTH{TH}
    toeplitzhankel::TH
end

function *(P::ChebyshevToLegendrePlanTH, v::AbstractVector{S}) where S
    n = length(v)
    ret = zero(S)
    @inbounds for k = 1:2:n
        ret += -v[k]/(k*(k-2))
    end
    v[1] = ret
    P.toeplitzhankel*view(v,2:n)
    v
end

function _cheb2leg_rescale1!(V::AbstractMatrix{S}) where S
    m,n = size(V)
    for j = 1:n
        ret = zero(S)
        @inbounds for k = 1:2:m
            ret += -V[k,j]/(k*(k-2))
        end
        V[1,j] = ret
    end
    V
end


function *(P::ChebyshevToLegendrePlanTH, V::AbstractMatrix)
    m,n = size(V)
    dims = P.toeplitzhankel.dims
    if dims == (1,)
        _cheb2leg_rescale1!(V)
        P.toeplitzhankel*view(V,2:m,:)
    elseif dims == (2,)
        _cheb2leg_rescale1!(transpose(V))
        P.toeplitzhankel*view(V,:,2:n)
    else
        @assert dims == (1,2)
        (R1,R2),(L1,L2),(T1,T2),tmp = P.toeplitzhankel.R,P.toeplitzhankel.L,P.toeplitzhankel.T,P.toeplitzhankel.tmp

        _cheb2leg_rescale1!(V)
        _th_applymul1!(view(V,2:m,:), T1, L1, R1, tmp)
        _cheb2leg_rescale1!(transpose(V))
        _th_applymul2!(view(V,:,2:n), T2, L2, R2, tmp)
    end
    V
end



function _leg2chebTH_TLC(::Type{S}, mn, d) where S
    n = mn[d]
    λ = Λ.(0:half(real(S)):n-1)
    t = zeros(S,n)
    t[1:2:end] .= 2 .* view(λ, 1:2:n) ./ π
    C = hankel_partialchol(λ)
    T = plan_uppertoeplitz!(t, (mn..., size(C,2)), d)
    L = copy(C)
    L[1,:] ./= 2
    T,L,C
end

function _leg2chebuTH_TLC(::Type{S}, mn, d) where {S}
    n = mn[d]
    S̃ = real(S)
    λ = Λ.(0:half(S̃):n-1)
    t = zeros(S,n)
    t[1:2:end] = λ[1:2:n]./(((1:2:n).-2))
    h = λ./((1:2n-1).+1)
    C = hankel_partialchol(h)
    T = plan_uppertoeplitz!(-2t/π, (length(t), size(C,2)), 1)
    (T, (1:n) .* C, C)
end


for f in (:leg2cheb, :leg2chebu)
    plan = Symbol("plan_th_", f, "!")
    TLC = Symbol("_", f, "TH_TLC")
    @eval begin
        $plan(::Type{S}, mn::Tuple, dims::Int) where {S} = ToeplitzHankelPlan($TLC(S, mn, dims)..., dims)

        function $plan(::Type{S}, mn::NTuple{2,Int}, dims::NTuple{2,Int}) where {S}
            @assert dims == (1,2)
            T1,L1,C1 = $TLC(S, mn, 1)
            T2,L2,C2 = $TLC(S, mn, 2)
            ToeplitzHankelPlan((T1,T2), (L1,L2), (C1,C2), dims)
        end
    end
end

_sub_dim_by_one(d) = ()
_sub_dim_by_one(d, m, n...) = (isone(d) ? m-1 : m, _sub_dim_by_one(d-1, n...)...)

function _cheb2legTH_TLC(::Type{S}, mn, d) where S
    n = mn[d]
    t = zeros(S,n-1)
    S̃ = real(S)
    if n > 1
        t[1:2:end] = Λ.(0:one(S̃):div(n-2,2), -half(S̃), one(S̃))
    end
    h = Λ.(1:half(S̃):n-1, zero(S̃), 3half(S̃))
    D = 1:n-1
    DL = (3half(S̃):n-half(S̃)) ./ D
    DR = -(one(S̃):n-one(S̃)) ./ (4 .* D)
    C = hankel_partialchol(h, D)
    T = plan_uppertoeplitz!(t, (_sub_dim_by_one(d, mn...)..., size(C,2)), d)
    T, DL .* C, DR .* C
end

plan_th_cheb2leg!(::Type{S}, mn::Tuple, dims::Int) where {S} = ChebyshevToLegendrePlanTH(ToeplitzHankelPlan(_cheb2legTH_TLC(S, mn, dims)..., dims))

function plan_th_cheb2leg!(::Type{S}, mn::NTuple{2,Int}, dims::NTuple{2,Int}) where {S}
    @assert dims == (1,2)
    T1,L1,C1 = _cheb2legTH_TLC(S, mn, 1)
    T2,L2,C2 = _cheb2legTH_TLC(S, mn, 2)
    ChebyshevToLegendrePlanTH(ToeplitzHankelPlan((T1,T2), (L1,L2), (C1,C2), dims))
end

function plan_th_ultra2ultra!(::Type{S}, (n,)::Tuple{Int}, λ₁, λ₂) where {S}
    @assert abs(λ₁-λ₂) < 1
    S̃ = real(S)
    DL = (zero(S̃):n-one(S̃)) .+ λ₂
    jk = 0:half(S̃):n-1
    t = zeros(S,n)
    t[1:2:n] = Λ.(jk,λ₁-λ₂,one(S̃))[1:2:n]
    h = Λ.(jk,λ₁,λ₂+one(S̃))
    lmul!(gamma(λ₂)/gamma(λ₁),h)
    C = hankel_partialchol(h)
    T = plan_uppertoeplitz!(lmul!(inv(gamma(λ₁-λ₂)),t), (length(t), size(C,2)), 1)
    ToeplitzHankelPlan(T, DL .* C, C)
end

function alternatesign!(v)
    @inbounds for k = 2:2:length(v)
        v[k] = -v[k]
    end
    v
end

function plan_th_jac2jac!(::Type{S}, (n,), α, β, γ, δ) where {S}
    if β == δ
        @assert abs(α-γ) < 1
        @assert α+β > -1
        jk = 0:n-1
        DL = (2jk .+ γ .+ β .+ 1).*Λ.(jk,γ+β+1,β+1)
        t = convert(AbstractVector{S}, Λ.(jk, α-γ,1))
        h = Λ.(0:2n-2,α+β+1,γ+β+2)
        DR = Λ.(jk,β+1,α+β+1)./gamma(α-γ)
        C = hankel_partialchol(h)
        T = plan_uppertoeplitz!(t, (length(t), size(C,2)), 1)
    elseif α == γ
        jk = 0:n-1
        DL = (2jk .+ δ .+ α .+ 1).*Λ.(jk,δ+α+1,α+1)
        h = Λ.(0:2n-2,α+β+1,δ+α+2)
        DR = Λ.(jk,α+1,α+β+1)./gamma(β-δ)
        t = alternatesign!(convert(AbstractVector{S}, Λ.(jk,β-δ,1)))
        C = hankel_partialchol(h)
        T = plan_uppertoeplitz!(t, (length(t), size(C,2)), 1)
    else
        throw(ArgumentError("Cannot create Toeplitz dot Hankel, use a sequence of plans."))
    end

    ToeplitzHankelPlan(T, DL .* C, DR .* C)
end

for f in (:th_leg2cheb, :th_cheb2leg, :th_leg2chebu)
    plan = Symbol("plan_", f, "!")
    @eval begin
        $plan(::Type{S}, mn::NTuple{N,Int}, dims::UnitRange) where {N,S} = $plan(S, mn, tuple(dims...))
        $plan(::Type{S}, mn::Tuple{Int}, dims::Tuple{Int}=(1,)) where {S} = $plan(S, mn, dims...)
        $plan(::Type{S}, (m,n)::NTuple{2,Int}) where {S} = $plan(S, (m,n), (1,2))
        $plan(arr::AbstractArray{T}, dims...) where T = $plan(T, size(arr), dims...)
        $f(v, dims...) = $plan(eltype(v), size(v), dims...)*copy(v)
    end
end

th_ultra2ultra(v, λ₁, λ₂, dims...) = plan_th_ultra2ultra!(eltype(v),size(v),λ₁,λ₂, dims...)*copy(v)
th_jac2jac(v, α, β, γ, δ, dims...) = plan_th_jac2jac!(eltype(v),size(v),α,β,γ,δ, dims...)*copy(v)
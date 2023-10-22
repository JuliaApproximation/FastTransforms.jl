"""
Represent a scaled Toeplitz∘Hankel matrix:

    DL(T∘H)DR

where the Hankel matrix `H` is non-negative definite, via

    ∑_{k=1}^r Diagonal(L[:,k])*T*Diagonal(R[:,k])

where `L` and `R` are determined by doing a rank-r pivoted Cholesky decomposition of `H`, which in low rank form is

    H ≈ ∑_{k=1}^r C[:,k]C[:,k]'

so that `L[:,k] = DL*C[:,k]` and `R[:,k] = DR*C[:,k]`.

This allows a Cholesky decomposition in 𝒪(K²N) operations and 𝒪(KN) storage, K = log N log ɛ⁻¹.
The tuple storage allows plans applied to each dimension.
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
    C = Matrix{T}(undef, n, 100)
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
    r == 100 && error("ranks more than 100 not yet supported")
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
    T = plan_uppertoeplitz!(-2t/π, (mn..., size(C,2)), d)
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

###
# th_cheb2leg
###

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


###
# th_ultra2ultra
###

# The second case handles zero
isapproxinteger(::Integer) = true
isapproxinteger(x) = isinteger(x) || x ≈ round(Int,x)  || x+1 ≈ round(Int,x+1)

"""
  _nearest_jacobi_par(α, γ)

returns a number that is an integer different than γ but less than 1 away from α.
"""
function _nearest_jacobi_par(α::T, γ::T) where T
    ret = isapproxinteger(α-γ) ? α : round(Int,α,RoundDown) + mod(γ,1)
    ret ≤ -1 ? ret + 1 : ret
end
_nearest_jacobi_par(α::T, ::T) where T<:Integer = α
_nearest_jacobi_par(α, γ) = _nearest_jacobi_par(promote(α,γ)...)


struct Ultra2UltraPlanTH{T, Plans, Dims} <: Plan{T}
    plans::Plans
    λ₁::T
    λ₂::T
    dims::Dims
end

function *(P::Ultra2UltraPlanTH, A::AbstractArray)
    ret = A
    if isapproxinteger(P.λ₂ - P.λ₁)
        _ultra2ultra_integerinc!(ret, P.λ₁, P.λ₂, P.dims)
    else
        for p in P.plans
            ret = p*ret
        end
        c = _nearest_jacobi_par(P.λ₁, P.λ₂)

        _ultra2ultra_integerinc!(ret, c, P.λ₂, P.dims)
    end
end

function _ultra2ultraTH_TLC(::Type{S}, mn, λ₁, λ₂, d) where {S}
    n = mn[d]
    @assert abs(λ₁-λ₂) < 1
    S̃ = real(S)
    DL = (zero(S̃):n-one(S̃)) .+ λ₂
    jk = 0:half(S̃):n-1
    t = zeros(S,n)
    t[1:2:n] = Λ.(jk,λ₁-λ₂,one(S̃))[1:2:n]
    h = Λ.(jk,λ₁,λ₂+one(S̃))
    lmul!(gamma(λ₂)/gamma(λ₁),h)
    C = hankel_partialchol(h)
    T = plan_uppertoeplitz!(lmul!(inv(gamma(λ₁-λ₂)),t), (mn..., size(C,2)), d)
    T, DL .* C, C
end

_good_plan_th_ultra2ultra!(::Type{S}, mn, λ₁, λ₂, dims::Int) where S = ToeplitzHankelPlan(_ultra2ultraTH_TLC(S, mn, λ₁, λ₂, dims)..., dims)

function _good_plan_th_ultra2ultra!(::Type{S}, mn::NTuple{2,Int}, λ₁, λ₂, dims::NTuple{2,Int}) where S
    T1,L1,C1 = _ultra2ultraTH_TLC(S, mn, λ₁, λ₂, 1)
    T2,L2,C2 = _ultra2ultraTH_TLC(S, mn, λ₁, λ₂, 2)
    ToeplitzHankelPlan((T1,T2), (L1,L2), (C1,C2), dims)
end



function plan_th_ultra2ultra!(::Type{S}, mn, λ₁, λ₂, dims) where {S}
    c = _nearest_jacobi_par(λ₁, λ₂)

    if isapproxinteger(λ₂ - λ₁)
        # TODO: don't make extra plan
        plans = typeof(_good_plan_th_ultra2ultra!(S, mn, λ₁+0.1, λ₂, dims))[]
    else
        plans = [_good_plan_th_ultra2ultra!(S, mn, λ₁, c, dims)]
    end

    Ultra2UltraPlanTH(plans, λ₁, λ₂, dims)
end

function _ultra_raise!(B, λ)
    m, n = size(B, 1), size(B, 2)

    if m > 1
        @inbounds for j = 1:n
            for i = 1:m-2
                Bij = λ / (i+λ-1) * B[i,j]
                Bij += -λ / (i+λ+1) * B[i+2,j]
                B[i,j] = Bij
            end
            B[m-1,j] = λ / (m+λ-2)*B[m-1,j]
            B[m,j] = λ / (m+λ-1)*B[m,j]
        end
    end
    B
end

function _ultra_lower!(B, λ)
    m, n = size(B, 1), size(B, 2)

    if m > 1
        @inbounds for j = 1:n
            B[m,j] = (m+λ-1)/λ * B[m,j]
            B[m-1,j] = (m+λ-2)/λ *B[m-1,j]
            for i = m-2:-1:1
                Bij = B[i,j] + λ / (i+λ+1) * B[i+2,j]
                B[i,j] = (i+λ-1)/λ * Bij
            end  
        end
    end
    B
end



function _ultra_raise!(x, λ, dims)
    for d in dims
        if d == 1
            _ultra_raise!(x, λ)
        else
            _ultra_raise!(x', λ)
        end
    end
    x
end

function _ultra_lower!(x, λ, dims)
    for d in dims
        if d == 1
            _ultra_lower!(x, λ-1)
        else
            _ultra_lower!(x', λ-1)
        end
    end
    x
end

function _ultra2ultra_integerinc!(x, λ₁, λ₂, dims)
    while !(λ₁ ≈ λ₂)
        if λ₂ > λ₁
            _ultra_raise!(x, λ₁, dims)
            λ₁ += 1
        else
            _ultra_lower!(x, λ₁, dims)
            λ₁ -= 1
        end
    end
    x
end

###
# th_jac2jac
###


function _lmul!(A::Bidiagonal, B::AbstractVecOrMat)
    @assert A.uplo == 'U'
    
    m, n = size(B, 1), size(B, 2)
    if m != size(A, 1)
        throw(DimensionMismatch("right hand side B needs first dimension of size $(size(A,1)), has size $m"))
    end
    @inbounds for j = 1:n
        for i = 1:m-1
            Bij = A.dv[i]*B[i,j]
            Bij += A.ev[i]*B[i+1,j]
            B[i,j] = Bij
        end
        B[m,j] = A.dv[m]*B[m,j]
    end
    B
end

struct Jac2JacPlanTH{T, Plans, Dims} <: Plan{T}
    plans::Plans
    α::T
    β::T
    γ::T
    δ::T
    dims::Dims
end

Jac2JacPlanTH(plans, α, β, γ, δ, dims) = Jac2JacPlanTH(plans, promote(α, β, γ, δ)..., dims)

function *(P::Jac2JacPlanTH, A::AbstractArray)
    if P.α + P.β ≤ -1
        _jacobi_raise_a!(A, P.α, P.β, P.dims)
        c,d = _nearest_jacobi_par(P.α+1, P.γ), _nearest_jacobi_par(P.β, P.δ)
    else
        c,d = _nearest_jacobi_par(P.α, P.γ), _nearest_jacobi_par(P.β, P.δ)
    end

    ret = A
    for p in P.plans
        ret = p*ret
    end

    _jac2jac_integerinc!(ret, c, d, P.γ, P.δ, P.dims)
end

function alternatesign!(v)
    @inbounds for k = 2:2:length(v)
        v[k] = -v[k]
    end
    v
end

function _jac2jacTH_TLC(::Type{S}, mn, α, β, γ, δ, d) where {S}
    n = mn[d]
    @assert α+β > -1
    if β == δ
        @assert abs(α-γ) < 1
        jk = 0:n-1
        DL = (2jk .+ γ .+ β .+ 1).*Λ.(jk,γ+β+1,β+1)
        t = convert(AbstractVector{S}, Λ.(jk, α-γ,1))
        h = Λ.(0:2n-2,α+β+1,γ+β+2)
        DR = Λ.(jk,β+1,α+β+1)./gamma(α-γ)
        C = hankel_partialchol(h)
        T = plan_uppertoeplitz!(t, (mn..., size(C,2)), d)
    elseif α == γ
        @assert abs(β-δ) < 1
        jk = 0:n-1
        DL = (2jk .+ δ .+ α .+ 1).*Λ.(jk,δ+α+1,α+1)
        h = Λ.(0:2n-2,α+β+1,δ+α+2)
        DR = Λ.(jk,α+1,α+β+1)./gamma(β-δ)
        t = alternatesign!(convert(AbstractVector{S}, Λ.(jk,β-δ,1)))
        C = hankel_partialchol(h)
        T = plan_uppertoeplitz!(t, (mn..., size(C,2)), d)
    else
        throw(ArgumentError("Cannot create Toeplitz dot Hankel, use a sequence of plans."))
    end

    (T, DL .* C, DR .* C)
end

_good_plan_th_jac2jac!(::Type{S}, mn, α, β, γ, δ, dims::Int) where S = ToeplitzHankelPlan(_jac2jacTH_TLC(S, mn, α, β, γ, δ, dims)..., dims)

function _good_plan_th_jac2jac!(::Type{S}, mn::NTuple{2,Int}, α, β, γ, δ, dims::NTuple{2,Int}) where S
    T1,L1,C1 = _jac2jacTH_TLC(S, mn, α, β, γ, δ, 1)
    T2,L2,C2 = _jac2jacTH_TLC(S, mn, α, β, γ, δ, 2)
    ToeplitzHankelPlan((T1,T2), (L1,L2), (C1,C2), dims)
end



function plan_th_jac2jac!(::Type{S}, mn, α, β, γ, δ, dims) where {S}
    if α + β ≤ -1
        c,d = _nearest_jacobi_par(α+1, γ), _nearest_jacobi_par(β, δ)
    else
        c,d = _nearest_jacobi_par(α, γ), _nearest_jacobi_par(β, δ)
    end

    if isapproxinteger(β - δ) && isapproxinteger(α-γ)
        # TODO: don't make extra plan
        plans = typeof(_good_plan_th_jac2jac!(S, mn, α+0.1, β, α, β, dims))[]
    elseif isapproxinteger(α - γ) || isapproxinteger(β - δ)
        if α + β ≤ -1
            # avoid degenerecies
            plans = [_good_plan_th_jac2jac!(S, mn, α+1, β, c, d, dims)]
        else
            plans = [_good_plan_th_jac2jac!(S, mn, α, β, c, d, dims)]
        end
    else
        if α + β ≤ -1
            plans = [_good_plan_th_jac2jac!(S, mn, α+1, β, α+1, d, dims), _good_plan_th_jac2jac!(S, mn, α+1, d, c, d, dims)]
        else
            plans = [_good_plan_th_jac2jac!(S, mn, α, β, α, d, dims), _good_plan_th_jac2jac!(S, mn, α, d, c, d, dims)]
        end
    end

    Jac2JacPlanTH(plans, α, β, γ, δ, dims)
end


function _jacobi_raise_a!(B, a, b)
    m, n = size(B, 1), size(B, 2)
    if m > 1
        @inbounds for j = 1:n
            B[1,j] = B[1,j] - (1+b) / (a+b+3) * B[2,j]
            for i = 2:m-1
                B[i,j] = (i+a+b)/(a+b-1+2i) * B[i,j] - (i+b) / (a+b+2i+1) * B[i+1,j]
            end
            B[m,j] = (m+a+b)/(a+b-1+2m)*B[m,j]
        end
    end
    B
end

function _jacobi_lower_a!(B, a, b)
    m, n = size(B, 1), size(B, 2)

    if m > 1
        @inbounds for j = 1:n
            B[m,j] = (a+b-1+2m)/(m+a+b) * B[m,j]
            for i = m-1:-1:2
                Bij = B[i,j] + (i+b) / (a+b+2i+1) * B[i+1,j]
                B[i,j] = (a+b-1+2i)/(i+a+b)  * Bij
            end
            B[1,j] = B[1,j] + (1+b) / (a+b+3) * B[2,j]
        end
    end
    B
end



function _jacobi_raise_b!(B, a, b)
    m, n = size(B, 1), size(B, 2)
    if m > 1
        @inbounds for j = 1:n
            B[1,j] = B[1,j] + (1+a) / (a+b+3) * B[2,j]
            
            for i = 2:m-1
                B[i,j] = (i+a+b)/(a+b-1+2i) * B[i,j] + (i+a) / (a+b+2i+1) * B[i+1,j]
            end
            B[m,j] = (m+a+b)/(a+b-1+2m)*B[m,j]
        end
    end
    B
end

function _jacobi_lower_b!(B, a, b)
    m, n = size(B, 1), size(B, 2)

    if m > 1
        @inbounds for j = 1:n
            B[m,j] = (a+b-1+2m)/(m+a+b) * B[m,j]
            for i = m-1:-1:2
                Bij = B[i,j] - (i+a) / (a+b+2i+1) * B[i+1,j]
                B[i,j] = (a+b-1+2i)/(i+a+b)  * Bij
            end
            B[1,j] = B[1,j] - (1+a) / (a+b+3) * B[2,j]
        end
    end
    B
end



function _jacobi_raise_b!(x, α, β, dims)
    for d in dims
        if d == 1
            _jacobi_raise_b!(x, α, β)
        else
            _jacobi_raise_b!(x', α, β)
        end
    end
    x
end
function _jacobi_raise_a!(x, α, β, dims)
    for d in dims
        if d == 1
            _jacobi_raise_a!(x, α, β)
        else
            _jacobi_raise_a!(x', α, β)
        end
    end
    x
end

function _jacobi_lower_b!(x, α, β, dims)
    for d in dims
        if d == 1
            _jacobi_lower_b!(x, α, β-1)
        else
            _jacobi_lower_b!(x', α, β-1)
        end
    end
    x
end
function _jacobi_lower_a!(x, α, β, dims)
    for d in dims
        if d == 1
            _jacobi_lower_a!(x, α-1, β)
        else
            _jacobi_lower_a!(x', α-1, β)
        end
    end
    x
end


function _jac2jac_integerinc!(x, α, β, γ, δ, dims)
    while !(α ≈ γ && β ≈ δ)
        if !(δ ≈ β) && δ > β
            _jacobi_raise_b!(x, α, β, dims)
            β += 1
        elseif !(δ ≈ β) && δ < β
            _jacobi_lower_b!(x, α, β, dims)
            β -= 1
        elseif !(γ ≈ α) && γ > α
            _jacobi_raise_a!(x, α, β, dims)
            α += 1
        else
            @assert γ < α
            _jacobi_lower_a!(x, α, β, dims)
            α -= 1
        end
    end
    x
end


###
# other routines
###

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

plan_th_ultra2ultra!(::Type{S}, mn::NTuple{N,Int}, λ₁, λ₂, dims::UnitRange) where {N,S} = plan_th_ultra2ultra!(S, mn, λ₁, λ₂, tuple(dims...))
plan_th_ultra2ultra!(::Type{S}, mn::Tuple{Int}, λ₁, λ₂, dims::Tuple{Int}=(1,)) where {S} = plan_th_ultra2ultra!(S, mn, λ₁, λ₂, dims...)
plan_th_ultra2ultra!(::Type{S}, (m,n)::NTuple{2,Int}, λ₁, λ₂) where {S} = plan_th_ultra2ultra!(S, (m,n), λ₁, λ₂, (1,2))
plan_th_ultra2ultra!(arr::AbstractArray{T}, λ₁, λ₂, dims...) where T = plan_th_ultra2ultra!(T, size(arr), λ₁, λ₂, dims...)
th_ultra2ultra(v, λ₁, λ₂, dims...) = plan_th_ultra2ultra!(eltype(v), size(v), λ₁, λ₂, dims...)*copy(v)

plan_th_jac2jac!(::Type{S}, mn::NTuple{N,Int}, α, β, γ, δ, dims::UnitRange) where {N,S} = plan_th_jac2jac!(S, mn, α, β, γ, δ, tuple(dims...))
plan_th_jac2jac!(::Type{S}, mn::Tuple{Int}, α, β, γ, δ, dims::Tuple{Int}=(1,)) where {S} = plan_th_jac2jac!(S, mn, α, β, γ, δ, dims...)
plan_th_jac2jac!(::Type{S}, (m,n)::NTuple{2,Int}, α, β, γ, δ) where {S} = plan_th_jac2jac!(S, (m,n), α, β, γ, δ, (1,2))
plan_th_jac2jac!(arr::AbstractArray{T}, α, β, γ, δ, dims...) where T = plan_th_jac2jac!(T, size(arr), α, β, γ, δ, dims...)
th_jac2jac(v, α, β, γ, δ, dims...) = plan_th_jac2jac!(eltype(v), size(v), α, β, γ, δ, dims...)*copy(v)


####
# cheb2jac
####

struct Cheb2JacPlanTH{T, Pl<:Jac2JacPlanTH{T}} <: Plan{T}
    jac2jac::Pl
end


struct Jac2ChebPlanTH{T, Pl<:Jac2JacPlanTH{T}} <: Plan{T}
    jac2jac::Pl
end


function jac_cheb_recurrencecoefficients(T, N)
    n = 0:N
    h = one(T)/2
    A = (2n .+ one(T)) ./ (n .+ one(T))
    A[1] /= 2
    A, Zeros(n), 
    ((n .- h) .* (n .- h) .* (2n .+ one(T))) ./ ((n .+ one(T)) .* n .* (2n .- one(T)))
end


function *(P::Cheb2JacPlanTH{T}, X::AbstractArray) where T
    A,B,C = jac_cheb_recurrencecoefficients(T, max(size(X)...))

    for d in P.jac2jac.dims
        if d == 1
            p = forwardrecurrence(size(X,1), A,B,C, one(T))
            X .= p .\ X
        else
            @assert d == 2
            n = size(X,2)
            p = forwardrecurrence(size(X,2), A,B,C, one(T))
            X .= X ./ transpose(p)
        end
    end
    P.jac2jac*X
end

function *(P::Jac2ChebPlanTH{T}, X::AbstractArray) where T
    X = P.jac2jac*X
    A,B,C = jac_cheb_recurrencecoefficients(T, max(size(X)...))

    for d in P.jac2jac.dims
        if d == 1
            p = forwardrecurrence(size(X,1), A,B,C, one(T))
            X .= p .* X
        else
            @assert d == 2
            n = size(X,2)
            p = forwardrecurrence(size(X,2), A,B,C, one(T))
            X .= X .* transpose(p)
        end
    end
    X
end

plan_th_cheb2jac!(::Type{T}, mn, α, β, dims...) where T = Cheb2JacPlanTH(plan_th_jac2jac!(T, mn, -one(α)/2, -one(α)/2, α, β, dims...))
plan_th_cheb2jac!(arr::AbstractArray{T}, α, β, dims...) where T = plan_th_cheb2jac!(T, size(arr), α, β, dims...)
th_cheb2jac(v, α, β, dims...) = plan_th_cheb2jac!(eltype(v), size(v), α, β, dims...)*copy(v)

plan_th_jac2cheb!(::Type{T}, mn, α, β, dims...) where T = Jac2ChebPlanTH(plan_th_jac2jac!(T, mn, α, β, -one(α)/2, -one(α)/2, dims...))
plan_th_jac2cheb!(arr::AbstractArray{T}, α, β, dims...) where T = plan_th_jac2cheb!(T, size(arr), α, β, dims...)
th_jac2cheb(v, α, β, dims...) = plan_th_jac2cheb!(eltype(v), size(v), α, β, dims...)*copy(v)
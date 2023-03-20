"""
Store a diagonally-scaled Toeplitz∘Hankel matrix:
    DL(T∘H)DR
where the Hankel matrix `H` is non-negative definite. This allows a Cholesky decomposition in 𝒪(K²N) operations and 𝒪(KN) storage, K = log N log ɛ⁻¹.
"""
struct ToeplitzHankelPlan{S, TP<:ToeplitzPlan} <: Plan{S}
    T::TP
    C::Vector{Vector{S}}
    DL::Vector{S}
    DR::Vector{S}
    ToeplitzHankelPlan{S,TP}(T::TP, C, DL, DR) where {S,TP} = new{S,TP}(T, C, DL, DR)
end

function ToeplitzHankelPlan(T::ToeplitzPlan, C::Vector, DL::AbstractVector, DR::AbstractVector)
    S = promote_type(eltype(T), eltype(C[1]), eltype(DL), eltype(DR))
    ToeplitzHankelPlan{S, typeof(T)}(T, C, collect(S,DL), collect(S,DR))
end
ToeplitzHankelPlan(T::ToeplitzPlan, C::Matrix) =
    ToeplitzHankelPlan(T, C, ones(size(T, 1)),ones(size(T,2)))
ToeplitzHankelPlan(T::ToeplitzPlan, H::Hankel, DL::AbstractVector, DR::AbstractVector) =
    ToeplitzHankelPlan(T, partialchol(H), DL, DR)
ToeplitzHankelPlan(T::ToeplitzPlan, H::Hankel, D::AbstractVector, DL::AbstractVector, DR::AbstractVector) =
    ToeplitzHankelPlan(T, partialchol(H,D), DL, DR)

*(P::ToeplitzHankelPlan, v::AbstractVector) = P.DL .* toeplitzcholmult(P.T, P.C, P.DR.*v)

function partialchol(H::Hankel)
    # Assumes positive definite
    σ = eltype(H)[]
    n = size(H,1)
    C = Vector{eltype(H)}[]
    v = [H[:,1]; vec(H[end,2:end])]
    d = diag(H)
    @assert length(v) ≥ 2n-1
    reltol = maximum(abs,d)*eps(eltype(H))*log(n)
    for k=1:n
        mx,idx = findmax(d)
        if mx ≤ reltol break end
        push!(σ, inv(mx))
        push!(C, v[idx:n+idx-1])
        for j=1:k-1
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

function partialchol(H::Hankel, D::AbstractVector)
    # Assumes positive definite
    T = promote_type(eltype(H),eltype(D))
    σ = T[]
    n = size(H,1)
    C = Vector{T}[]
    v = [H[:,1];vec(H[end,2:end])]
    d = diag(H).*D.^2
    @assert length(v) ≥ 2n-1
    reltol = maximum(abs,d)*eps(T)*log(n)
    for k = 1:n
        mx,idx = findmax(d)
        if mx ≤ reltol break end
        push!(σ,inv(mx))
        push!(C,v[idx:n+idx-1].*D.*D[idx])
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

function toeplitzcholmult(T, C, v)
    n,K = length(v),length(C)
    ret,temp1 = zero(v),zero(v)
    temp1 .= C[K] .* v
    T * temp1
    ret .= C[K] .* temp1
    for k = K-1:-1:1
        temp1 .= C[k] .* v
        T * temp1
        temp1 .= C[k] .* temp1
        ret .= ret .+ temp1
    end
    ret
end


# Diagonally-scaled Toeplitz∘Hankel polynomial transforms

function leg2chebTH(::Type{S}, n) where S
    λ = Λ.(0:half(S):n-1)
    t = zeros(S,n)
    t[1:2:end] = λ[1:2:n]
    T = plan_uppertoeplitz!(2t/π)
    H = Hankel(λ[1:n], λ[n:end])
    DL = ones(S,n)
    DL[1] /= 2
    T,H,DL
end

function cheb2legTH(::Type{S},n) where S
    t = zeros(S,n-1)
    t[1:2:end] = Λ.(0:one(S):div(n-2,2), -half(S), one(S))
    T = plan_uppertoeplitz!(t)
    h = Λ.(1:half(S):n-1, zero(S), 3half(S))
    H = Hankel(h[1:n-1],h[n-1:end])
    D = 1:one(S):n-1
    DL = (3half(S):n-half(S))./D
    DR = -(one(S):n-one(S))./4D
    T,H,D,DL,DR
end

function leg2chebuTH(::Type{S},n) where S
    λ = Λ.(0:half(S):n-1)
    t = zeros(S,n)
    t[1:2:end] = λ[1:2:n]./(((1:2:n).-2))
    T = plan_uppertoeplitz!(-2t/π)
    H = Hankel(λ[1:n]./((1:n).+1),λ[n:end]./((n:2n-1).+1))
    T,H
end

function ultra2ultraTH(::Type{S},n,λ₁,λ₂) where S
    @assert abs(λ₁-λ₂) < 1
    DL = (zero(S):n-one(S)) .+ λ₂
    jk = 0:half(S):n-1
    t = zeros(S,n)
    t[1:2:n] = Λ.(jk,λ₁-λ₂,one(S))[1:2:n]
    T = plan_uppertoeplitz!(lmul!(inv(gamma(λ₁-λ₂)),t))
    h = Λ.(jk,λ₁,λ₂+one(S))
    lmul!(gamma(λ₂)/gamma(λ₁),h)
    H = Hankel(h[1:n],h[n:end])
    DR = ones(S,n)
    T,H,DL,DR
end

function jac2jacTH(::Type{S},n,α,β,γ,δ) where S
    if β == δ
        @assert abs(α-γ) < 1
        @assert α+β > -1
        jk = zero(S):n-one(S)
        DL = (2jk .+ γ .+ β .+ one(S)).*Λ.(jk,γ+β+one(S),β+one(S))
        T = plan_uppertoeplitz!(Λ.(jk,α-γ,one(S)))
        H = Hankel(Λ.(jk,α+β+one(S),γ+β+two(S)),Λ.(jk.+n.-one(S),α+β+one(S),γ+β+two(S)))
        DR = Λ.(jk,β+one(S),α+β+one(S))./gamma(α-γ)
        T,H,DL,DR
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

plan_th_leg2cheb(::Type{S}, n) where {S} = ToeplitzHankelPlan(leg2chebTH(S, n)..., ones(S, n))
plan_th_cheb2leg(::Type{S}, n) where {S} = ChebyshevToLegendrePlanTH(ToeplitzHankelPlan(cheb2legTH(S, n)...))
plan_th_leg2chebu(::Type{S}, n) where {S} = ToeplitzHankelPlan(leg2chebuTH(S, n)..., 1:n, ones(S, n))
plan_th_ultra2ultra(::Type{S}, n, λ₁, λ₂) where {S} = ToeplitzHankelPlan(ultra2ultraTH(S, n, λ₁, λ₂)...)
plan_th_jac2jac(::Type{S},n, α, β, γ, δ) where {S} = ToeplitzHankelPlan(jac2jacTH(S, n, α, β, γ, δ)...)


th_leg2cheb(v) = plan_th_leg2cheb(eltype(v),length(v))*v
th_cheb2leg(v) = plan_th_cheb2leg(eltype(v),length(v))*v
th_leg2chebu(v) = plan_th_leg2chebu(eltype(v),length(v))*v
th_ultra2ultra(v,λ₁,λ₂) = plan_th_ultra2ultra(eltype(v),length(v),λ₁,λ₂)*v
th_jac2jac(v,α,β,γ,δ) = plan_th_jac2jac(eltype(v),length(v),α,β,γ,δ)*v
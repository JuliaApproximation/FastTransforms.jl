"""
Store a diagonally-scaled Toeplitz∘Hankel matrix:

    DL(T∘H)DR

where the Hankel matrix `H` is non-negative definite. This allows a Cholesky decomposition in 𝒪(K²N) operations and 𝒪(KN) storage, K = log N log ɛ⁻¹.
"""
immutable ToeplitzHankelPlan{S}
    T::TriangularToeplitz{S}
    C::Vector{Vector{S}}
    DL::Vector{S}
    DR::Vector{S}
    ToeplitzHankelPlan(T,C,DL,DR)=new(T,C,DL,DR)
end

function ToeplitzHankelPlan(T::TriangularToeplitz,C::Vector,DL::AbstractVector,DR::AbstractVector)
    S=promote_type(eltype(T),eltype(C[1]),eltype(DL),eltype(DR))
    ToeplitzHankelPlan{S}(T,C,collect(S,DL),collect(S,DR))
end
ToeplitzHankelPlan(T::TriangularToeplitz,C::Matrix) =
    ToeplitzHankelPlan(T,C,ones(size(T,1)),ones(size(T,2)))
ToeplitzHankelPlan(T::TriangularToeplitz,H::Hankel,D...) =
    ToeplitzHankelPlan(T,partialchol(H),D...)

*(P::ToeplitzHankelPlan,v::AbstractVector)=P.DL.*toeplitzcholmult(P.T,P.C,P.DR.*v)

function partialchol(H::Hankel)
    # Assumes positive definite
    σ=eltype(H)[]
    n=size(H,1)
    C=Vector{eltype(H)}[]
    v=[H[:,1];vec(H[end,2:end])]
    d=diag(H)
    @assert length(v) ≥ 2n-1
    reltol=maxabs(d)*eps(eltype(H))*log(n)
    for k=1:n
        mx,idx=findmax(d)
        if mx ≤ reltol break end
        push!(σ,inv(mx))
        push!(C,v[idx:n+idx-1])
        for j=1:k-1
            nCjidxσj = -C[j][idx]*σ[j]
            Base.axpy!(nCjidxσj, C[j], C[k])
        end
        @simd for p=1:n
            @inbounds d[p]-=C[k][p]^2/mx
        end
    end
    for k=1:length(σ) scale!(C[k],sqrt(σ[k])) end
    C
end

function toeplitzcholmult(T,C,v)
    n,K = length(v),length(C)
    ret,temp1,temp2 = zero(v),zero(v),zero(v)
    un,ze = one(eltype(v)),zero(eltype(v))
    broadcast!(*, temp1, C[K], v)
    A_mul_B!(un, T, temp1, ze, temp2)
    broadcast!(*, ret, C[K], temp2)
    for k=K-1:-1:1
        broadcast!(*, temp1, C[k], v)
        A_mul_B!(un, T, temp1, ze, temp2)
        broadcast!(*, temp1, C[k], temp2)
        broadcast!(+, ret, ret, temp1)
    end
    ret
end


# Diagonally-scaled Toeplitz∘Hankel polynomial transforms

function leg2chebTH{S}(::Type{S},n)
    λ = Λ(0:half(S):n-1)
    t = zeros(S,n)
    t[1:2:end] = λ[1:2:n]
    T = TriangularToeplitz(2t/π,:U)
    H = Hankel(λ[1:n],λ[n:end])
    DL = ones(S,n)
    DL[1] /= 2
    T,H,DL
end

function cheb2legTH{S}(::Type{S},n)
    t = zeros(S,n-1)
    t[1:2:end] = Λ(0:one(S):div(n-2,2),-half(S),one(S))
    T = TriangularToeplitz(t,:U)
    h = Λ(1:half(S):n-1,zero(S),3half(S))
    H = Hankel(h[1:n-1],h[n-1:end])
    DL = 3half(S):n-half(S)
    DR = -(one(S):n-one(S))/4
    T,H,DL,DR
end

function leg2chebuTH{S}(::Type{S},n)
    λ = Λ(0:half(S):n-1)
    t = zeros(S,n)
    t[1:2:end] = λ[1:2:n]./(((1:2:n)-2))
    T = TriangularToeplitz(-2t/π,:U)
    H = Hankel(λ[1:n]./((1:n)+1),λ[n:end]./((n:2n-1)+1))
    T,H
end

function ultra2ultraTH{S}(::Type{S},n,λ₁,λ₂)
    @assert abs(λ₁-λ₂) < 1
    DL = (zero(S):n-one(S))+λ₂
    jk = 0:half(S):n-1
    t = zeros(S,n)
    t[1:2:n] = Λ(jk,λ₁-λ₂,one(S))[1:2:n]
    T = TriangularToeplitz(scale!(inv(gamma(λ₁-λ₂)),t),:U)
    h = Λ(jk,λ₁,λ₂+one(S))
    scale!(gamma(λ₂)/gamma(λ₁),h)
    H = Hankel(h[1:n],h[n:end])
    DR = ones(S,n)
    T,H,DL,DR
end

function jac2jacTH{S}(::Type{S},n,α,β,γ,δ)
    @assert β == δ
    @assert abs(α-γ) < 1
    @assert α+β > -1
    jk = zero(S):n-one(S)
    DL = (2jk+γ+β+one(S)).*Λ(jk,γ+β+one(S),β+one(S))
    T = TriangularToeplitz(Λ(jk,α-γ,one(S)),:U)
    H = Hankel(Λ(jk,α+β+one(S),γ+β+two(S)),Λ(jk+n-one(S),α+β+one(S),γ+β+two(S)))
    DR = Λ(jk,β+one(S),α+β+one(S))/gamma(α-γ)
    T,H,DL,DR
end

th_leg2chebplan{S}(::Type{S},n)=ToeplitzHankelPlan(leg2chebTH(S,n)...,ones(S,n))
th_cheb2legplan{S}(::Type{S},n)=ToeplitzHankelPlan(cheb2legTH(S,n)...)
th_leg2chebuplan{S}(::Type{S},n)=ToeplitzHankelPlan(leg2chebuTH(S,n)...,1:n,ones(S,n))
th_ultra2ultraplan{S}(::Type{S},n,λ₁,λ₂)=ToeplitzHankelPlan(ultra2ultraTH(S,n,λ₁,λ₂)...)
th_jac2jacplan{S}(::Type{S},n,α,β,γ,δ)=ToeplitzHankelPlan(jac2jacTH(S,n,α,β,γ,δ)...)


immutable ChebyshevToLegendrePlan{TH}
    toeplitzhankel::TH
end

ChebyshevToLegendrePlan{S}(::Type{S},n) = ChebyshevToLegendrePlan(th_cheb2legplan(S,n))


function *(P::ChebyshevToLegendrePlan,v::AbstractVector)
    w = zero(v)
    S,n = eltype(v),length(v)
    w[1:2:end] = -one(S)./(one(S):two(S):n)./(-one(S):two(S):n-two(S))
    [dot(w,v);P.toeplitzhankel*view(v,2:n)]
end


th_leg2cheb(v)=th_leg2chebplan(eltype(v),length(v))*v
th_cheb2leg(v) = ChebyshevToLegendrePlan(eltype(v),length(v))*v

th_leg2chebu(v)=th_leg2chebuplan(eltype(v),length(v))*v
th_ultra2ultra(v,λ₁,λ₂)=th_ultra2ultraplan(eltype(v),length(v),λ₁,λ₂)*v
th_jac2jac(v,α,β,γ,δ)=th_jac2jacplan(eltype(v),length(v),α,β,γ,δ)*v

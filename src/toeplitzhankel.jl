function partialchol(H::Hankel)
    # Assumes positive definite
    σ=eltype(H)[]
    n=size(H,1)
    C=Vector{eltype(H)}[]
    v=[H[:,1];vec(H[end,2:end])]
    d=diag(H)
    @assert length(v) ≥ 2n-1
    tol=eps(eltype(H))*log(n)
    for k=1:n
        mx,idx=findmax(d)
        if mx ≤ tol break end
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


# Plan a multiply by DL*(T.*H)*DR
immutable ToeplitzHankelPlan{S}
    T::TriangularToeplitz{S}
    C::Vector{Vector{S}}   # A cholesky factorization of H: H=CC'
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

*(P::ToeplitzHankelPlan,v::Vector)=P.DL.*toeplitzcholmult(P.T,P.C,P.DR.*v)



# Legendre transforms

function leg2chebTH{S}(::Type{S},n)
    λ=Λ(0:half(S):n-1)
    t=zeros(S,n)
    t[1:2:end]=λ[1:2:n]
    T=TriangularToeplitz(2/π*t,:U)
    H=Hankel(λ[1:n],λ[n:end])
    DL=ones(S,n)
    DL[1]/=2
    T,H,DL
end

function leg2chebuTH{S}(::Type{S},n)
    λ=Λ(0:half(S):n-1)
    t=zeros(S,n)
    t[1:2:end]=λ[1:2:n]./(((1:2:n)-2))
    T=TriangularToeplitz(-2/π*t,:U)
    H=Hankel(λ[1:n]./((1:n)+1),λ[n:end]./((n:2n-1)+1))
    T,H
end

function jac2jacTH{S}(::Type{S},n,α,β,γ,δ)
    @assert β == δ
    jk = zero(S):n-one(S)
    DL = (2jk+γ+β+one(S)).*Λ(jk,γ+β+one(S),β+one(S))
    T = TriangularToeplitz(Λ(jk,α-γ,one(S)),:U)
    H = Hankel(Λ(jk,α+β+one(S),γ+β+two(S)),Λ(jk+n-1,α+β+one(S),γ+β+two(S)))
    DR = Λ(jk,β+one(S),α+β+one(S))/gamma(α-γ)
    T,H,DL,DR
end

th_leg2chebplan{S}(::Type{S},n)=ToeplitzHankelPlan(leg2chebTH(S,n)...,ones(S,n))
th_leg2chebuplan{S}(::Type{S},n)=ToeplitzHankelPlan(leg2chebuTH(S,n)...,1:n,ones(S,n))
th_jac2jacplan{S}(::Type{S},n,α,β,γ,δ)=ToeplitzHankelPlan(jac2jacTH(S,n,α,β,γ,δ)...)

th_leg2cheb(v)=th_leg2chebplan(eltype(v),length(v))*v
th_leg2chebu(v)=th_leg2chebuplan(eltype(v),length(v))*v
th_jac2jac(v,α,β,γ,δ)=th_jac2jacplan(eltype(v),length(v),α,β,γ,δ)*v
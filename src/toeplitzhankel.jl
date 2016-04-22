function partialchol(H::Hankel)
    # Assumes positive definite
    σ=Array(eltype(H),0)
    n=size(H,1)
    C=Array(eltype(H),n,n)
    v=[H[:,1];vec(H[end,2:end])]
    d=diag(H)

    @assert length(v) ≥ 2n-1

    tol=1E-14
    for k=1:n
        mx,idx=findmax(d)
        if mx ≤ tol
            break
        end
        push!(σ,1/mx)
        @simd for p=1:n
            @inbounds C[p,k]= v[p+idx-1]             # H[p,idx]
        end
        for j=1:k-1,p=1:n
            @inbounds C[p,k]-=C[p,j]*C[idx,j]*σ[j]
        end
        @simd for p=1:n
            @inbounds d[p]-=C[p,k]^2/mx
        end
    end
    for k=1:length(σ),p=1:n
        @inbounds C[p,k]*=sqrt(σ[k])
    end
    C[:,1:length(σ)]
end

function toeplitzcholmult(T,C,v)
    ret=C[:,1].*(T*(C[:,1].*v))
    for j=2:size(C,2)
        ret+=C[:,j].*(T*(C[:,j].*v))
    end
    ret
end


# Plan a multiply by DL*(T.*H)*DR
immutable ToepltizHankelPlan{T}
    T::TriangularToeplitz{T}
    C::Matrix{T}   # A cholesky factorization of H: H=CC'
    DL::Vector{T}
    DR::Vector{T}
end

ToepltizHankelPlan(T::TriangularToeplitz,C::Matrix)=ToepltizHankelPlan(T,C,ones(size(T,1)),ones(size(T,2)))
ToepltizHankelPlan(T::TriangularToeplitz,H::Hankel,D...)=ToepltizHankelPlan(T,partialchol(H),D...)

*(P::ToepltizHankelPlan,v::Vector)=P.DL.*toeplitzcholmult(P.T,P.C,P.DR.*v)



# Legendre transforms


Λ{T}(::Type{T},z)=z<5?gamma(z+one(T)/2)/gamma(z+one(T)):exp(lgamma(z+one(T)/2)-lgamma(z+one(T)))

# use recurrence to construct Λ fast on a range of values
function Λ{T}(::Type{T},r::UnitRange)
    n=length(r)
    ret=Array(T,n)
    ret[1]=Λ(T,r[1])
    for k=2:n
        ret[k]=ret[k-1]*(r[k-1]+one(T)/2)/r[k]
    end
    ret
end

function Λ{T}(::Type{T},r::FloatRange)
    @assert step(r)==0.5
    n=length(r)
    ret=Array(T,n)
    ret[1]=Λ(T,r[1])
    ret[2]=Λ(T,r[2])
    for k=3:n
        ret[k]=ret[k-2]*(r[k-2]+one(T)/2)/r[k]
    end
    ret
end

Λ(z::Number)=Λ(typeof(z),z)
Λ(z::AbstractArray)=Λ(eltype(z),z)


function leg2chebuTH(n)
    λ=Λ(0:0.5:n-1)
    t=zeros(n)
    t[1:2:end]=λ[1:2:n]./(((1:2:n)-2))
    T=TriangularToeplitz(-2/π*t,:U)
    H=Hankel(λ[1:n]./((1:n)+1),λ[n:end]./((n:2n-1)+1))
    T,H
end

leg2chebuplan(n)=ToepltizHankelPlan(leg2chebuTH(n)...,1:n,ones(n))

function leg2chebTH{TT}(::Type{TT},n)
    λ=Λ(TT,0:0.5:n-1)
    t=zeros(TT,n)
    t[1:2:end]=λ[1:2:n]
    T=TriangularToeplitz(2/π*t,:U)
    H=Hankel(λ[1:n],λ[n:end])
    DL=ones(TT,n)
    DL[1]/=2
    T,H,DL
end

leg2chebplan{TT}(::Type{TT},n)=ToepltizHankelPlan(leg2chebTH(TT,n)...,ones(TT,n))

function leg2chebTHslow(n)
    λ=map(Λ,0:0.5:n-1)
    t=zeros(n)
    t[1:2:end]=λ[1:2:n]
    T=TriangularToeplitz(2/π*t,:U)
    H=Hankel(λ[1:n],λ[n:end])
    DL=ones(n)
    DL[1]*=0.5
    T,H,DL
end


leg2cheb(v)=leg2chebplan(eltype(v),length(v))*v
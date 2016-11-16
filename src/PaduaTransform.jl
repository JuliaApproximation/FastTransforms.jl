"""
Pre-plan an Inverse Padua Transform.
"""
immutable IPaduaTransformPlan{IDCTPLAN,T}
    cfsmat::Matrix{T}
    padvals::Vector{T}
    idctplan::IDCTPLAN
end

function plan_ipaduatransform{T}(v::AbstractVector{T})
    N=length(v)
    n=Int(cld(-3+sqrt(1+8N),2))
    @assert N==div((n+1)*(n+2),2)
    IPaduaTransformPlan(Array{T}(n+2,n+1),Array{T}(N),
        FFTW.plan_r2r!(Array{T}(n+2,n+1),FFTW.REDFT00))
end

"""
Inverse Padua Transform maps the 2D Chebyshev coefficients to the values of the interpolation polynomial at the Padua points.
"""
function *{T}(P::IPaduaTransformPlan,v::AbstractVector{T})
    cfsmat=trianglecfsmat(P,v)
    n,m=size(cfsmat)
    scale!(view(cfsmat,:,2:m-1),0.5)
    scale!(view(cfsmat,2:n-1,:),0.5)
    tensorvals=P.idctplan*cfsmat
    paduavals=paduavec(P,tensorvals)
    return paduavals
end

ipaduatransform(v::AbstractVector) = plan_ipaduatransform(v)*v

"""
Creates (n+2)x(n+1) Chebyshev coefficient matrix from triangle coefficients.
"""
function trianglecfsmat(P::IPaduaTransformPlan,cfs::AbstractVector)
    N=length(cfs)
    n=Int(cld(-3+sqrt(1+8N),2))
    cfsmat=fill!(P.cfsmat,0)
    m=1
    for d=1:n+1
        @inbounds for k=1:d
            j=d-k+1
            cfsmat[k,j]=cfs[m]
            if m==N
                return cfsmat
            else
                m+=1
            end
        end
    end
    return cfsmat
end

"""
Vectorizes the function values at the Padua points.
"""
function paduavec(P::IPaduaTransformPlan,padmat::Matrix)
    n=size(padmat,2)-1
    N=(n+1)*(n+2)
    if iseven(n)>0
        d=div(n+2,2)
        m=0
        @inbounds for i=1:n+1
            P.padvals[m+1:m+d]=view(padmat,1+mod(i,2):2:n+1+mod(i,2),i)
            m+=d
        end
    else
        @inbounds P.padvals[:]=view(padmat,1:2:N-1)
    end
    return P.padvals
end

"""
Pre-plan a Padua Transform.
"""
immutable PaduaTransformPlan{DCTPLAN,T}
    vals::Matrix{T}
    retvec::Vector{T}
    dctplan::DCTPLAN
end

function plan_paduatransform{T}(v::AbstractVector{T})
    N=length(v)
    n=Int(cld(-3+sqrt(1+8N),2))
    @assert N==div((n+1)*(n+2),2)
    PaduaTransformPlan(Array{T}(n+2,n+1),Array{T}(N),
        FFTW.plan_r2r!(Array{T}(n+2,n+1),FFTW.REDFT00))
end

"""
Padua Transform maps from interpolant values at the Padua points to the 2D Chebyshev coefficients.
"""
function *{T}(P::PaduaTransformPlan,v::AbstractVector{T})
    N=length(v)
    n=Int(cld(-3+sqrt(1+8N),2))
    vals=paduavalsmat(P,v)
    tensorcfs=P.dctplan*vals
    m,l=size(tensorcfs)
    scale!(tensorcfs,T(2)/(n*(n+1)))
    scale!(view(tensorcfs,1,:),0.5)
    scale!(view(tensorcfs,:,1),0.5)
    scale!(view(tensorcfs,m,:),0.5)
    scale!(view(tensorcfs,:,l),0.5)
    cfs=trianglecfsvec(P,tensorcfs)
    return cfs
end

paduatransform(v::AbstractVector) = plan_paduatransform(v)*v

"""
Creates (n+2)x(n+1) matrix of interpolant values on the tensor grid at the (n+1)(n+2)/2 Padua points.
"""
function paduavalsmat(P::PaduaTransformPlan,v::AbstractVector)
    N=length(v)
    n=Int(cld(-3+sqrt(1+8N),2))
    vals=fill!(P.vals,0.)
    if iseven(n)>0
        d=div(n+2,2)
        m=0
        @inbounds for i=1:n+1
            vals[1+mod(i,2):2:n+1+mod(i,2),i]=view(v,m+1:m+d)
            m+=d
        end
    else
        @inbounds vals[1:2:end]=view(v,:)
    end
    return vals
end

"""
Creates length (n+1)(n+2)/2 vector from matrix of triangle Chebyshev coefficients.
"""
function trianglecfsvec(P::PaduaTransformPlan,cfs::Matrix)
    m=size(cfs,2)
    l=1
    for d=1:m
        @inbounds for k=1:d
            j=d-k+1
            P.retvec[l]=cfs[k,j]
            l+=1
        end
    end
    return P.retvec
end

"""
Returns coordinates of the (n+1)(n+2)/2 Padua points.
"""
function paduapoints{T}(::Type{T},n::Integer)
    N=div((n+1)*(n+2),2)
    MM=Array(T,N,2)
    m=0
    delta=0
    NN=fld(n+2,2)
    @inbounds for k=n:-1:0
        if isodd(n)>0
            delta=mod(k,2)
        end
        @inbounds for j=NN+delta:-1:1
            m+=1
            MM[m,1]=sinpi(T(k)/n-T(0.5))
            if isodd(n-k)>0
                MM[m,2]=sinpi((2j-one(T))/(n+1)-T(0.5))
            else
                MM[m,2]=sinpi(T(2j-2)/(n+1)-T(0.5))
            end
        end
    end
    return MM
end

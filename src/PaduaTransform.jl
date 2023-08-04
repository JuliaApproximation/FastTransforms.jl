
# lex indicates if its lexigraphical (i.e., x, y) or reverse (y, x)
# If in lexigraphical order the coefficient vector's entries
# corrrespond to the following basis polynomials:
# [T0(x) * T0(y), T1(x) * T0(y), T0(x) * T1(y), T2(x) * T0(y), T1(x) * T1(y), T0(x) * T2(y), ...]
# else, if not in lexigraphical order:
# [T0(x) * T0(y), T0(x) * T1(y), T1(x) * T0(y), T0(x) * T2(y), T1(x) * T1(y), T2(x) * T0(y), ...]
"""
Pre-plan an Inverse Padua Transform.
"""
struct IPaduaTransformPlan{lex,IDCTPLAN,T}
    cfsmat::Matrix{T}
    idctplan::IDCTPLAN
end

IPaduaTransformPlan(cfsmat::Matrix{T},idctplan,::Type{Val{lex}}) where {T,lex} =
    IPaduaTransformPlan{lex,typeof(idctplan),T}(cfsmat,idctplan)

"""
Pre-plan an Inverse Padua Transform.
"""
function plan_ipaduatransform!(::Type{T},N::Integer,lex) where T
    n=Int(cld(-3+sqrt(1+8N),2))
    if N ≠ div((n+1)*(n+2),2)
        error("Padua transforms can only be applied to vectors of length (n+1)*(n+2)/2.")
    end
    IPaduaTransformPlan(Array{T}(undef,n+2,n+1),FFTW.plan_r2r!(Array{T}(undef,n+2,n+1),FFTW.REDFT00),lex)
end


plan_ipaduatransform!(::Type{T},N::Integer) where {T} = plan_ipaduatransform!(T,N,Val{true})
plan_ipaduatransform!(v::AbstractVector{T},lex...) where {T} = plan_ipaduatransform!(eltype(v),length(v),lex...)


function *(P::IPaduaTransformPlan,v::AbstractVector{T}) where T
    cfsmat=trianglecfsmat(P,v)
    n,m=size(cfsmat)
    rmul!(view(cfsmat,:,2:m-1),0.5)
    rmul!(view(cfsmat,2:n-1,:),0.5)
    tensorvals=P.idctplan*cfsmat
    paduavec!(v,P,tensorvals)
end

ipaduatransform!(v::AbstractVector,lex...) = plan_ipaduatransform!(v,lex...)*v
"""
Inverse Padua Transform maps the 2D Chebyshev coefficients to the values of the interpolation polynomial at the Padua points.
"""
ipaduatransform(v::AbstractVector,lex...) = plan_ipaduatransform!(v,lex...)*copy(v)

"""
Creates ``(n+2)x(n+1)`` Chebyshev coefficient matrix from triangle coefficients.
"""
function trianglecfsmat(P::IPaduaTransformPlan{true},cfs::AbstractVector)
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

function trianglecfsmat(P::IPaduaTransformPlan{false},cfs::AbstractVector)
    N=length(cfs)
    n=Int(cld(-3+sqrt(1+8N),2))
    cfsmat=fill!(P.cfsmat,0)
    m=1
    for d=1:n+1
        @inbounds for k=d:-1:1
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
function paduavec!(v,P::IPaduaTransformPlan,padmat::Matrix)
    n=size(padmat,2)-1
    N=(n+1)*(n+2)
    if iseven(n)>0
        d=div(n+2,2)
        m=0
        @inbounds for i=1:n+1
            v[m+1:m+d]=view(padmat,1+mod(i,2):2:n+1+mod(i,2),i)
            m+=d
        end
    else
        @inbounds v[:]=view(padmat,1:2:N-1)
    end
    return v
end

"""
Pre-plan a Padua Transform.
"""
struct PaduaTransformPlan{lex,DCTPLAN,T}
    vals::Matrix{T}
    dctplan::DCTPLAN
end

PaduaTransformPlan(vals::Matrix{T},dctplan,::Type{Val{lex}}) where {T,lex} =
    PaduaTransformPlan{lex,typeof(dctplan),T}(vals,dctplan)

"""
Pre-plan a Padua Transform.
"""
function plan_paduatransform!(::Type{T},N::Integer,lex) where T
    n=Int(cld(-3+sqrt(1+8N),2))
    if N ≠ ((n+1)*(n+2))÷2
        error("Padua transforms can only be applied to vectors of length (n+1)*(n+2)/2.")
    end
    PaduaTransformPlan(Array{T}(undef,n+2,n+1),FFTW.plan_r2r!(Array{T}(undef,n+2,n+1),FFTW.REDFT00),lex)
end

plan_paduatransform!(::Type{T},N::Integer) where {T} = plan_paduatransform!(T,N,Val{true})
plan_paduatransform!(v::AbstractVector{T},lex...) where {T} = plan_paduatransform!(eltype(v),length(v),lex...)

function *(P::PaduaTransformPlan,v::AbstractVector{T}) where T
    N=length(v)
    n=Int(cld(-3+sqrt(1+8N),2))
    vals=paduavalsmat(P,v)
    tensorcfs=P.dctplan*vals
    m,l=size(tensorcfs)
    rmul!(tensorcfs,T(2)/(n*(n+1)))
    rmul!(view(tensorcfs,1,:),0.5)
    rmul!(view(tensorcfs,:,1),0.5)
    rmul!(view(tensorcfs,m,:),0.5)
    rmul!(view(tensorcfs,:,l),0.5)
    trianglecfsvec!(v,P,tensorcfs)
end

paduatransform!(v::AbstractVector,lex...) = plan_paduatransform!(v,lex...)*v
"""
Padua Transform maps from interpolant values at the Padua points to the 2D Chebyshev coefficients.
"""
paduatransform(v::AbstractVector,lex...) = plan_paduatransform!(v,lex...)*copy(v)

"""
Creates ``(n+2)x(n+1)`` matrix of interpolant values on the tensor grid at the ``(n+1)(n+2)/2`` Padua points.
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
Creates length ``(n+1)(n+2)/2`` vector from matrix of triangle Chebyshev coefficients.
"""
function trianglecfsvec!(v,P::PaduaTransformPlan{true},cfs::Matrix)
    m=size(cfs,2)
    l=1
    for d=1:m
        @inbounds for k=1:d
            j=d-k+1
            v[l]=cfs[k,j]
            l+=1
        end
    end
    return v
end

function trianglecfsvec!(v,P::PaduaTransformPlan{false},cfs::Matrix)
    m=size(cfs,2)
    l=1
    for d=1:m
        @inbounds for k=d:-1:1
            j=d-k+1
            v[l]=cfs[k,j]
            l+=1
        end
    end
    return v
end

"""
Returns coordinates of the ``(n+1)(n+2)/2`` Padua points.
"""
function paduapoints(::Type{T}, n::Integer) where T
    N=div((n+1)*(n+2),2)
    MM=Matrix{T}(undef,N,2)
    m=0
    delta=0
    NN=div(n,2)+1
    # x coordinates
    for k=n:-1:0
        if isodd(n)
            delta = Int(isodd(k))
        end
        x = -cospi(T(k)/n)
        @inbounds for j=NN+delta:-1:1
            m+=1
            MM[m,1]=x
        end
    end
    # y coordinates
    # populate the first two sets, and copy the rest
    m=0
    for k=n:-1:n-1
        if isodd(n)
            delta = Int(isodd(k))
        end
        for j=NN+delta:-1:1
            m+=1
            @inbounds if isodd(n-k)
                MM[m,2]=-cospi((2j-one(T))/(n+1))
            else
                MM[m,2]=-cospi(T(2j-2)/(n+1))
            end
        end
    end
    m += 1
    # number of y coordinates between k=n and k=n-2
    Ny_shift = 2NN+isodd(n)
    for k in n-2:-1:0
        if isodd(n)
            delta = Int(isodd(k))
        end
        for j in range(m, length=NN+delta)
            @inbounds MM[j,2] = MM[j-Ny_shift,2]
        end
        m += NN+delta
    end
    return MM
end

paduapoints(n::Integer) = paduapoints(Float64,n)

"""
Pre-plan an Inverse Padua Transform.
"""
immutable IPaduaTransformPlan{DCTPLAN}
    cfsmat::Matrix{Float64}
    tensorvals::Matrix{Float64}
    padvals::Vector{Float64}
    dctplan::DCTPLAN
end

function plan_ipaduatransform(v::AbstractVector)
    N=length(v)
    n=Int(cld(-3+sqrt(1+8N),2))
    @assert N==div((n+1)*(n+2),2)
    IPaduaTransformPlan(zeros(Float64,n+2,n+1),Array(Float64,n+2,n+1),
        Array(Float64,N),FFTW.plan_r2r(Array(eltype(v),n+2,n+1),FFTW.REDFT00))
end

"""
Inverse Padua Transform maps the 2D Chebyshev coefficients to the values of the interpolation polynomial at the Padua points.
"""
function ipaduatransform(P::IPaduaTransformPlan,v::AbstractVector)
    cfsmat=trianglecfsmat(P,v)
    tensorvals=P.tensorvals
    cfsmat[:,2:end-1]=scale!(cfsmat[:,2:end-1],0.5)
    cfsmat[2:end-1,:]=scale!(cfsmat[2:end-1,:],0.5)
    tensorvals=P.dctplan*cfsmat
    paduavals=paduavec(P,tensorvals)
    return paduavals
end

function ipaduatransform(v::AbstractVector)
    cfsmat=trianglecfsmat(v)
    n=size(cfsmat,2)-1
    tensorvals=Array(Float64,n+2,n+1)
    cfsmat[:,2:end-1]=scale!(cfsmat[:,2:end-1],0.5)
    cfsmat[2:end-1,:]=scale!(cfsmat[2:end-1,:],0.5)
    tensorvals= FFTW.r2r(cfsmat,FFTW.REDFT00)
    paduavals=paduavec(tensorvals)
    return paduavals
end
"""
Creates (n+2)x(n+1) Chebyshev coefficient matrix from triangle coefficients.
"""
function trianglecfsmat(P::IPaduaTransformPlan,cfs::AbstractVector)
    N=length(cfs)
    n=Int(cld(-3+sqrt(1+8N),2))
    cfsmat=P.cfsmat
    m=1
    for d=1:n+1, k=1:d
        j=d-k+1
        cfsmat[k,j]=cfs[m]
        if m==N
            return cfsmat
        else
            m+=1
        end
    end
    return cfsmat
end

function trianglecfsmat(cfs::AbstractVector)
    N=length(cfs)
    n=Int(cld(-3+sqrt(1+8N),2))
    @assert N==div((n+1)*(n+2),2)
    cfsmat=zeros(Float64,n+2,n+1)
    m=1
    for d=1:n+1, k=1:d
        j=d-k+1
        cfsmat[k,j]=cfs[m]
        if m==N
            return cfsmat
        else
            m+=1
        end
    end
    return cfsmat
end

"""
Vectorizes the function values at the Padua points.
"""
function paduavec(P::IPaduaTransformPlan,padmat::Matrix)
    n=size(padmat,2)-1
    padvals=P.padvals
    if iseven(n)>0
        d=div(n+2,2)
        m=0
        for i=1:n+1
            padvals[m+1:m+d]=padmat[1+mod(i,2):2:end-1+mod(i,2),i]
            m+=d
        end
    else
        padvals=padmat[1:2:end-1]
    end
    return padvals
end

function paduavec(padmat::Matrix)
    n=size(padmat,2)-1
    N=div((n+1)*(n+2),2)
    padvals=Array(Float64,N)
    if iseven(n)>0
        d=div(n+2,2)
        m=0
        for i=1:n+1
            padvals[m+1:m+d]=padmat[1+mod(i,2):2:end-1+mod(i,2),i]
            m+=d
        end
    else
        padvals=padmat[1:2:end-1]
    end
    return padvals
end

"""
Pre-plan a Padua Transform.
"""
immutable PaduaTransformPlan{IDCTPLAN}
    vals::Matrix{Float64}
    tensorcfs::Matrix{Float64}
    retvec::Vector{Float64}
    idctplan::IDCTPLAN
end

function plan_paduatransform(v::AbstractVector)
    N=length(v)
    n=Int(cld(-3+sqrt(1+8N),2))
    @assert N==div((n+1)*(n+2),2)
    PaduaTransformPlan(zeros(Float64,n+2,n+1),Array(Float64,n+2,n+1),
        zeros(Float64,N),FFTW.plan_r2r(Array(eltype(v),n+2,n+1),FFTW.REDFT00))
end

"""
Padua Transform maps from interpolant values at the Padua points to the 2D Chebyshev coefficients.
"""
function paduatransform(P::PaduaTransformPlan,v::AbstractVector)
    N=length(v)
    n=Int(cld(-3+sqrt(1+8N),2))
    tensorcfs=P.tensorcfs
    vals=paduavalsmat(P,v)
    tensorcfs=P.idctplan*vals
    tensorcfs=scale!(tensorcfs,2./(n*(n+1.)))
    tensorcfs[1,:]=scale!(tensorcfs[1,:],0.5)
    tensorcfs[:,1]=scale!(tensorcfs[:,1],0.5)
    tensorcfs[end,:]=scale!(tensorcfs[end,:],0.5)
    tensorcfs[:,end]=scale!(tensorcfs[:,end],0.5)
    cfs=trianglecfsvec(P,tensorcfs)
    return cfs
end

function paduatransform(v::AbstractVector)
    N=length(v)
    n=Int(cld(-3+sqrt(1+8N),2))
    tensorcfs=Array(Float64,n+2,n+1)
    vals=paduavalsmat(v)
    tensorcfs=FFTW.r2r(vals,FFTW.REDFT00)
    tensorcfs=scale!(tensorcfs,2./(n*(n+1.)))
    tensorcfs[1,:]=scale!(tensorcfs[1,:],0.5)
    tensorcfs[:,1]=scale!(tensorcfs[:,1],0.5)
    tensorcfs[end,:]=scale!(tensorcfs[end,:],0.5)
    tensorcfs[:,end]=scale!(tensorcfs[:,end],0.5)
    cfsvec=trianglecfsvec(tensorcfs)
    return cfsvec
end

"""
Creates (n+2)x(n+1) matrix of interpolant values on the tensor grid at the (n+1)(n+2)/2 Padua points.
"""
function paduavalsmat(P::PaduaTransformPlan,v::AbstractVector)
    N=length(v)
    n=Int(cld(-3+sqrt(1+8N),2))
    vals=P.vals
    if iseven(n)>0
        d=div(n+2,2)
        m=0
        for i=1:n+1
            vals[1+mod(i,2):2:end-1+mod(i,2),i]=v[m+1:m+d]
            m+=d
        end
    else
       vals[1:2:end]=v
    end
    return vals
end

function paduavalsmat(v::AbstractVector)
    N=length(v)
    n=Int(cld(-3+sqrt(1+8N),2))
    @assert N==div((n+1)*(n+2),2)
    vals=zeros(Float64,n+2,n+1)
    if iseven(n)>0
        d=div(n+2,2)
        m=0
        for i=1:n+1
            vals[1+mod(i,2):2:end-1+mod(i,2),i]=v[m+1:m+d]
            m+=d
        end
    else
       vals[1:2:end]=v
    end
    return vals
end

"""
Creates length (n+1)(n+2)/2 vector from matrix of triangle Chebyshev coefficients.
"""
function trianglecfsvec(P::PaduaTransformPlan,cfs::Matrix)
    m=size(cfs,2)
    ret=P.retvec
    l=1
    for d=1:m,k=1:d
        j=d-k+1
        ret[l]=cfs[k,j]
        l+=1
    end
    return ret
end

function trianglecfsvec(cfs::Matrix)
    n,m=size(cfs)
    N=div(n*m,2)
    ret=Array(Float64,N)
    l=1
    for d=1:m,k=1:d
        j=d-k+1
        ret[l]=cfs[k,j]
        l+=1
    end
    return ret
end

"""
Returns coordinates of the (n+1)(n+2)/2 Padua points.
"""
function paduapoints(n::Integer)
    N=div((n+1)*(n+2),2)
    MM=Array(Float64,N,2)
    m=0
    delta=0
    NN=fld(n+2,2)
    for k=n:-1:0
        if isodd(n)>0
            delta=mod(k,2)
        end
        for j=NN+delta:-1:1
            m+=1
            MM[m,1]=sinpi(1.*k/n-0.5)
            if isodd(n-k)>0
                MM[m,2]=sinpi((2j-1.)/(n+1.)-0.5)
            else
                MM[m,2]=sinpi((2j-2.)/(n+1.)-0.5)
            end
        end
    end
    return MM
end

"""
Interpolates a 2d function at a given point using 2d Chebyshev series.
"""
function paduaeval(f::Function,x::Float64,y::Float64,m::Integer)
    M=div((m+1)*(m+2),2)
    pvals=Array(Float64,M)
    p=paduapoints(m)
    pvals=map(f,p[:,1],p[:,2])
    plan=plan_paduatransform(pvals)
    coeffs=paduatransform(plan,pvals)
    cfs_mat=trianglecfsmat(coeffs)
    cfs_mat=cfs_mat[1:end-1,:]
    f_x=sum([cfs_mat[k,j]*cos((j-1)*acos(x))*cos((k-1)*acos(y)) for k=1:m+1, j=1:m+1])
    return f_x
end

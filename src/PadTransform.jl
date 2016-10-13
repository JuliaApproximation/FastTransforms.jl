
# Creates (n+2)x(n+1) Chebyshev coefficient matrix from triangle coefficients.
function trianglecfs(cfs::AbstractVector)
    N=length(cfs)
    n=ceil(Int,(-3+sqrt(1+8N))/2)
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

function paduavec(padmat::Matrix)
    n=size(padmat,2)-1
    N=div((n+1)*(n+2),2)
    padvals=zeros(N)
     if iseven(n)>0
        padmat=[padmat; zeros(1,n+1)]
        indices=zeros(div(n,2))
        indices=(n+3)*collect(1:div(n,2))
        padvals=padmat[2:2:end]
        padvals=deleteat!(padvals,indices)
    else
        padvals=padmat[1:2:end-1]
    end
    return padvals
end

# Inverse Padua Transform maps the 2D Chebyshev coefficients to the values of the interpolation polynomial at the Padua points.
function ipadtransform(cfs::AbstractVector)

    cfsmat=trianglecfs(cfs)
    n=size(cfsmat,2)-1
    tensorvals=zeros(Float64,n+2,n+1)

    cfsmat[:,2:end-1]=cfsmat[:,2:end-1]/2
    cfsmat[2:end-1,:]=cfsmat[2:end-1,:]/2
    tensorvals=FFTW.r2r(cfsmat,FFTW.REDFT00)
    paduavals=paduavec(tensorvals)

    return paduavals
end

# Creates (n+2)x(n+1) matrix of interpolant values on the tensor grid at the (n+1)*(n+2)/2 Padua points.
function padvalsmat(v::AbstractVector)
    N=length(v)
    n=ceil(Int,(-3+sqrt(1+8N))/2)
    @assert N==div((n+1)*(n+2),2)
    vals=zeros(Float64,n+2,n+1)

    if iseven(n)>0
       vals[2:2:end]=v
       vals=[vals; zeros(1,n+1)]
       indices=zeros(n)
       indices=sort([(n+3)*collect(1:2:n+1); (n+4)+(n+3)*collect(0:2:n-1)])
       vals=reshape(deleteat!(vec(vals),indices),n+2,n+1)
    else
       vals[1:2:end]=v
    end
    return vals
end

# Creates length (n+1)*(n+2)/2 vector from matrix of triangle Chebyshev coefficients.
function cfsvec(cfs::Matrix)
    n,m=size(cfs)
    ret=Array(Float64,sum(1:m))
    n=1
    for d=1:m,k=1:d
        j=d-k+1
        ret[n]=cfs[k,j]
        n+=1
    end
    return ret
end

# Padua Transform maps from interpolant values at the Padua points to the 2D Chebyshev coefficients.
function padtransform(v::AbstractVector)

        N=length(v)
        n=ceil(Int,(-3+sqrt(1+8N))/2)
        vals=padvalsmat(v)
        tensorcfs=zeros(Float64,n+2,n+1)

        tensorcfs=FFTW.r2r(vals,FFTW.REDFT00)
        tensorcfs=tensorcfs*2/(n*(n+1))
        tensorcfs[1,:]=tensorcfs[1,:]/2
        tensorcfs[:,1]=tensorcfs[:,1]/2
        tensorcfs[end,:]=tensorcfs[end,:]/2
        tensorcfs[:,end]=tensorcfs[:,end]/2
        cfs=cfsvec(tensorcfs)

    return cfs
end

# Returns coordinates of the (n+1)*(n+2)/2 Padua points
function padpoints(m::Integer)
  M=div((m+1)*(m+2),2)
  θ=linspace(0.,pi,m+2)
  ϕ=linspace(0.,pi,m+1)
  pts=zeros(M,2)
  X=zeros(Float64,m+2,m+1)
  Y=zeros(Float64,m+2,m+1)
  X=[cos(ϕ) for θ in θ, ϕ in ϕ]
  Y=[cos(θ) for θ in θ, ϕ in ϕ]
    if iseven(m)>0
        X=[X; zeros(1,m+1)]
        Y=[Y; zeros(1,m+1)]
        indices=zeros(div(m,2))
        indices=(m+3)*collect(1:div(m,2))
        pts[:,1]=deleteat!(X[2:2:end],indices)
        pts[:,2]=deleteat!(Y[2:2:end],indices)
    else
        pts[:,1]=X[1:2:end-1]
        pts[:,2]=Y[1:2:end-1]
    end
  return pts
end

# Interpolates a 2d function at a given point using 2d Chebyshev series
function padeval(f::Function,x::Float64,y::Float64,m::Integer)

    p=padpoints(m)
    pvals=zeros(div((m+1)*(m+2),2))
    pvals=map(f,p[:,1],p[:,2])
    coeffs=padtransform(pvals)
    cfs_mat=trianglecfs(coeffs)
    cfs_mat=cfs_mat[1:end-1,:]
    f_x=sum([cfs_mat[k,j]*cos((j-1)*acos(x))*cos((k-1)*acos(y)) for k=1:size(cfs_mat,1), j=1:size(cfs_mat,2)])

    return f_x
end

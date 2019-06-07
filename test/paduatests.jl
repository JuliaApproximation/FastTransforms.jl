using FastTransforms, Compat
using Compat.Test

@testset "Padua transform and its inverse" begin
    n=200
    N=div((n+1)*(n+2),2)
    v=rand(N)  #Length of v is the no. of Padua points
    Pl=plan_paduatransform!(v)
    IPl=plan_ipaduatransform!(v)
    @test Pl*(IPl*copy(v)) ≈ v
    @test IPl*(Pl*copy(v)) ≈ v
    @test Pl*copy(v) ≈ paduatransform(v)
    @test IPl*copy(v) ≈ ipaduatransform(v)

    # check that the return vector is NOT reused
    Pl=plan_paduatransform!(v)
    x=Pl*v
    y=Pl*rand(N)
    @test x ≠ y

    IPl=plan_ipaduatransform!(v)
    x=IPl*v
    y=IPl*rand(N)
    @test x ≠ y

    println("Testing runtimes for (I)Padua Transforms")
    @time Pl*v
    @time IPl*v

    println("Accuracy of 2d function interpolation at a point")

    """
    Interpolates a 2d function at a given point using 2d Chebyshev series.
    """
    function paduaeval(f::Function,x::AbstractFloat,y::AbstractFloat,m::Integer,lex)
        T=promote_type(typeof(x),typeof(y))
        M=div((m+1)*(m+2),2)
        pvals=Vector{T}(undef,M)
        p=paduapoints(T,m)
        map!(f,pvals,p[:,1],p[:,2])
        coeffs=paduatransform(pvals,lex)
        plan=plan_ipaduatransform!(pvals,lex)
        cfs_mat=FastTransforms.trianglecfsmat(plan,coeffs)
        f_x=sum([cfs_mat[k,j]*cos((j-1)*acos(x))*cos((k-1)*acos(y)) for k=1:m+1, j=1:m+1])
        return f_x
    end
    f_xy = (x,y) ->x^2*y+x^3
    g_xy = (x,y) ->cos(exp(2*x+y))*sin(y)
    x=0.1;y=0.2
    m=130
    l=80
    f_m=paduaeval(f_xy,x,y,m,Val{true})
    g_l=paduaeval(g_xy,x,y,l,Val{true})
    @test f_xy(x,y) ≈ f_m
    @test g_xy(x,y) ≈ g_l

    f_m=paduaeval(f_xy,x,y,m,Val{false})
    g_l=paduaeval(g_xy,x,y,l,Val{false})
    @test f_xy(x,y) ≈ f_m
    @test g_xy(x,y) ≈ g_l
end

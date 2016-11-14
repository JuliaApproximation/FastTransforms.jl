using FastTransforms
using Base.Test

println("Testing special functions")
n = 0:1000_000
λ = 0.123
@time FastTransforms.Cnλ(n,λ);
@time FastTransforms.Cnλ(n,λ);

x = linspace(0,20,81);
@test norm((FastTransforms.Λ(x)-FastTransforms.Λ(big(x)))./FastTransforms.Λ(big(x)),Inf) < 2eps()

x = 0:0.5:10_000
λ₁,λ₂ = 0.125,0.875
@test norm((FastTransforms.Λ(x,λ₁,λ₂)-FastTransforms.Λ(big(x),big(λ₁),big(λ₂)))./FastTransforms.Λ(big(x),big(λ₁),big(λ₂)),Inf) < 4eps()
λ₁,λ₂ = 1//3,2//3
@test norm((FastTransforms.Λ(x,Float64(λ₁),Float64(λ₂))-FastTransforms.Λ(big(x),big(λ₁),big(λ₂)))./FastTransforms.Λ(big(x),big(λ₁),big(λ₂)),Inf) < 4eps()

n = 0:1000
α = 0.125
β = 0.375

@time FastTransforms.Cnαβ(n,α,β);
@test norm(FastTransforms.Cnαβ(n,α,β)./FastTransforms.Cnαβ(n,big(α),big(β))-1,Inf) < 3eps()

@time FastTransforms.Anαβ(n,α,β);
@test norm(FastTransforms.Anαβ(n,α,β)./FastTransforms.Anαβ(n,big(α),big(β))-1,Inf) < 4eps()


N = 20
f(x) = exp(x)

x,w = FastTransforms.fejer1(N,0.,0.)
@test norm(dot(f(x),w)-2sinh(1)) ≤ 4eps()
x,w = FastTransforms.fejer2(N,0.,0.)
@test norm(dot(f(x),w)-2sinh(1)) ≤ 4eps()
x,w = FastTransforms.clenshawcurtis(N,0.,0.)
@test norm(dot(f(x),w)-2sinh(1)) ≤ 4eps()

#=
x = Fun(identity)
g = f(x)*(1-x)^.25*(1+x)^.35
val = sum(g)
=#

x,w = FastTransforms.fejer1(N,0.25,0.35)
@test norm(dot(f(x),w)-2.0351088204147243) ≤ 4eps()
x,w = FastTransforms.fejer2(N,0.25,0.35)
@test norm(dot(f(x),w)-2.0351088204147243) ≤ 4eps()
x,w = FastTransforms.clenshawcurtis(N,0.25,0.35)
@test norm(dot(f(x),w)-2.0351088204147243) ≤ 4eps()

println("Testing the Chebyshev–Jacobi transform")

Nr = 5
v = zeros(Nr)
Na,Nb = 5,5
V = zeros(Na,Nb)

for N in round(Int,logspace(1,3,3))
    println("")
    println("N = ",N)
    println("")
    for αi=1:Na, βi=1:Nb
        α,β = -0.45+(αi-1)/Na,-0.45+(βi-1)/Nb
        p1,p2 = plan_cjt(rand(N),α,β),plan_icjt(rand(N),α,β)
        for i=1:Nr
            c = rand(N)
            v[i] = log(norm(p2*(p1*c)-c,Inf)/(20N^(1+2max(α,β))*eps()))
        end
        mean(v) > 2 && println("This is the mean log(||Error||_∞/Estimate) that broke the test: ",mean(v)," and the standard deviation: ",std(v))
        V[αi,βi] = mean(v)
    end
    println("This is the mean log(||Error||_∞/Estimate): ",mean(V)," and the standard deviation: ",std(V))
end

println("Testing the special cases length(c) = 0,1,2:")

c = [1.5]
@test cjt(c,0.12,0.34) == c
@test icjt(c,0.12,0.34) == c

pop!(c)

@test cjt(c,0.12,0.34) == c
@test icjt(c,0.12,0.34) == c

c = [1.0;2.0]
@test norm(jjt(c,0.12,0.34,0.12,0.34)-c,Inf) ≤ 2eps()


# test same parameters

c=[0.,1.]
@test jjt(c,1.5,-0.5,1.5,-0.5) == c

println("Testing the special cases (α,β) = (±0.5,±0.5)")

# Chebyshev coefficients of exp
c_cheb = [1.2660658777520084,1.13031820798497,0.27149533953407656,0.044336849848663804,0.005474240442093732,0.0005429263119139438,4.497732295429515e-5,3.1984364624019905e-6,1.9921248066727958e-7,1.1036771725517344e-8,5.505896079673747e-10,2.4979566169849825e-11,1.03915223067857e-12,3.9912633564144015e-14,1.4237580108256572e-15,4.740926102561496e-17,1.4801800572082976e-18,4.3499194949441696e-20,1.2074289272797528e-21,3.175356737059445e-23,7.93367197163804e-25,1.8879484042289162e-26,4.2886738765925875e-28,9.318985281777577e-30,1.9406469749017399e-31,3.8798022492260015e-33,7.45850288739152e-35,1.3807477824110646e-36,2.4648623717710964e-38,4.2485421925059915e-40,7.079001176212895e-42,1.141486778254089e-43,1.783151037543284e-45,2.7011422638738024e-47,3.9714338657415316e-49,5.672351695612112e-51,7.876788130533199e-53,1.0642416282087358e-54,1.4000817885528905e-56,1.794689085390544e-58,2.243019482662972e-60,2.7349926016641967e-62,3.2554929833033215e-64,3.78495689402033e-66,4.300544446926769e-68,4.77780574839433e-70,5.192665133046628e-72,5.523442810988436e-74,5.717985746408847e-76]

# Jacobi coefficients of exp at (±0.5,±0.5)
c_00 = [1.2660658777520084,2.26063641596994,0.723987572090871,0.14187791951572415,0.020020079331085648,0.0022061767595233264,0.00019937999439479753,1.5268992715616027e-5,1.014420290055232e-6,5.9506859054175615e-8,3.124851418974203e-9,1.4852160704995775e-10,6.447143371527455e-12,2.5753240175931934e-13,9.526907336854026e-15,3.281725133829881e-16,1.0576496993856426e-17,3.20238467156949e-19,9.14299140846687e-21,2.4694550834939014e-22,6.3281713790321685e-24,1.542622093731937e-25,3.5857224408995815e-27,7.964666406806272e-29,1.6939043515668796e-30,3.4556190450066486e-32,6.773312266576111e-34,1.2775611026088504e-35,2.3221236341654026e-37,4.072731101981327e-39,6.901079409575277e-41,1.1310395206213411e-42,1.794875997727078e-44,2.760732681473006e-46,4.119631242417262e-48,5.969295899580336e-50,8.405881613439669e-52,1.1512859570474932e-53,1.5347890791678689e-55,1.9929132418968355e-57,2.522290550037395e-59,3.1134869971034086e-61,3.7506700259829663e-63,4.411970068315684e-65,5.0705898146983576e-67,5.696604001638848e-69,6.259278760298182e-71,6.729592140384465e-73,7.039950057933299e-75]
c_01 = [1.8312249817444934,1.4018135475190467,0.4211095858436539,0.07969774446521205,0.011002819207328322,0.0011944708771926122,0.00010677917238679192,8.110005868771403e-6,5.353106062476433e-7,3.1237733951100556e-8,1.6333110219427631e-9,7.735005972383578e-11,3.3473853404557083e-12,1.333595312027857e-13,4.92207038322879e-15,1.6920924742289326e-16,5.443658341283791e-18,1.6456374329092366e-19,4.691719175403533e-21,1.2655773772197326e-22,3.2393803393291905e-24,7.888321906112733e-26,1.8318188278743866e-27,4.065263937281931e-29,8.638847091039724e-31,1.7610248038105726e-32,3.449351261471638e-34,6.50183837007916e-36,1.1810743750838164e-37,2.070295858087742e-39,3.506179552173043e-41,5.743539156119835e-43,9.110325158556408e-45,1.4006615828866467e-46,2.0892357224279913e-48,3.026093616078655e-50,4.2597272086552854e-52,5.832159509538512e-54,7.772313549445791e-56,1.0089104305392274e-57,1.2765228632360985e-59,1.5752735945134057e-61,1.8971383534453687e-63,2.2310498815373167e-65,2.563461449357282e-67,2.8792582164274203e-69,3.1629293838222696e-71,3.399629156416382e-73,3.5199750289666496e-75]
c_10 = [0.7009067737595234,0.8588228684508934,0.30287798624721707,0.062180175050512106,0.009017260123757326,0.0010117058823307142,9.260082200800561e-5,7.1589868468446235e-6,4.791096838075886e-7,2.8269125103075062e-8,1.49154039703144e-9,7.117154732612197e-11,3.099758031071747e-12,1.2417287055653363e-13,4.604836953625235e-15,1.5896326596009483e-16,5.1328386525726346e-18,1.5567472386602532e-19,4.451272233063337e-21,1.2038777062741688e-22,3.088791039702978e-24,7.537899031206638e-26,1.753903613025195e-27,3.899402469524341e-29,8.300196424629072e-31,1.694594241196076e-32,3.3239610051044727e-34,6.273772656009344e-36,1.1410492590815863e-37,2.002435243893585e-39,3.3948998574022335e-41,5.566856050093576e-43,8.838434818714371e-45,1.3600710985863591e-46,2.0303955199892708e-48,2.943202283501681e-50,4.146154404784383e-52,5.68070006093642e-54,7.575577242232898e-56,9.84002811357608e-58,1.2457676868012966e-59,1.5382134025900027e-61,1.8535316725375976e-63,2.1809201867783675e-65,2.5071283653410757e-67,2.8173457852114277e-69,3.0963493764759127e-71,3.3299629839680836e-73,3.5199750289666496e-75]
c_11 = [1.13031820798497,0.7239875720908708,0.21281687927358628,0.040040158662171296,0.005515441898808317,0.0005981399831843923,5.34414745046561e-5,4.057681160220927e-6,2.677808657437903e-7,1.5624257094871016e-8,8.168688387747676e-10,3.8682860229164727e-11,1.6739606114355753e-12,6.668835135797819e-14,2.4612938503724105e-15,8.46119759508514e-17,2.722026970834067e-18,8.228692267620182e-20,2.3459823293192055e-21,6.328171379032168e-23,1.619753198418534e-24,3.944294684989539e-26,9.159366367827211e-28,2.0326852218802552e-29,4.31952380625831e-31,8.805305946548944e-33,1.7247074885219479e-34,3.2509730878315635e-36,5.905460097872925e-38,1.0351619114362916e-39,1.7531112569630786e-41,2.8718015963633246e-43,4.5552089244304594e-45,7.003373112109346e-47,1.044626782426559e-48,1.5130586904191403e-50,2.129879020537863e-52,2.916099250418951e-54,3.8861808216988285e-56,5.04458110007479e-58,6.382648344061989e-60,7.876407054564243e-62,9.485735646882059e-64,1.1155297592778338e-65,1.2817359024131178e-67,1.4396345613848593e-69,1.5814715428162693e-71,1.700107488097128e-73,1.7781317156635653e-75]


@test norm(cjt(c_00,-0.5,-0.5)-c_cheb,Inf) < eps()
@test norm(cjt(c_01,-0.5,0.5)-c_cheb,Inf) < eps()
@test norm(cjt(c_10,0.5,-0.5)-c_cheb,Inf) < eps()
@test norm(cjt(c_11,0.5,0.5)-c_cheb,Inf) < eps()

@test norm(icjt(c_cheb,-0.5,-0.5)-c_00,Inf) < eps()
@test norm(icjt(c_cheb,-0.5,0.5)-c_01,Inf) < eps()
@test norm(icjt(c_cheb,0.5,-0.5)-c_10,Inf) < eps()
@test norm(icjt(c_cheb,0.5,0.5)-c_11,Inf) < eps()


c = exp(-collect(1:1000)./30)

println("Testing increment/decrement operators for α,β ≤ -0.5")

α,β = -0.9,-0.6

@test norm(FastTransforms.fromsquare!(FastTransforms.tosquare!(copy(c),α,β),α,β)-c,Inf) < 10eps()
@test norm(icjt(cjt(c,α,β),α,β)-c,Inf) < 2000eps()
p1,p2 = plan_cjt(c,α,β),plan_icjt(c,α,β)
@test norm(p2*(p1*c)-c,Inf) < 2000eps()

println("Testing increment/decrement operators for α ≤ -0.5, β > -0.5")

α,β = -0.9,3.1

@test norm(FastTransforms.fromsquare!(FastTransforms.tosquare!(copy(c),α,β),α,β)-c,Inf) < 10eps()
@test norm(icjt(cjt(c,α,β),α,β)-c,Inf) < 2000eps()
p1,p2 = plan_cjt(c,α,β),plan_icjt(c,α,β)
@test norm(p2*(p1*c)-c,Inf) < 2000eps()

println("Testing increment/decrement operators for α > -0.5, β ≤ -0.5")

α,β = 2.4,-0.75

@test norm(FastTransforms.fromsquare!(FastTransforms.tosquare!(copy(c),α,β),α,β)-c,Inf) < 1200eps()
@test norm(icjt(cjt(c,α,β),α,β)-c,Inf) < 1e6eps()
p1,p2 = plan_cjt(c,α,β),plan_icjt(c,α,β)
@test norm(p2*(p1*c)-c,Inf) < 1e6eps()

println("Testing increment/decrement operators for α > -0.5, β > -0.5")

α,β = 2.4,3.6

@test norm(FastTransforms.fromsquare!(FastTransforms.tosquare!(copy(c),α,β),α,β)-c,Inf) < 250eps()
@test norm(icjt(cjt(c,α,β),α,β)-c,Inf) < 2e5eps()
p1,p2 = plan_cjt(c,α,β),plan_icjt(c,α,β)
@test norm(p2*(p1*c)-c,Inf) < 2e5eps()

println("Testing for complex coefficients")

α,β = 0.12,0.34
c = complex(rand(100),rand(100))

@test cjt(c,α,β) == complex(cjt(real(c),α,β),cjt(imag(c),α,β))
@test icjt(c,α,β) == complex(icjt(real(c),α,β),icjt(imag(c),α,β))
@test jjt(c,α,β,α,β) == complex(jjt(real(c),α,β,α,β),jjt(imag(c),α,β,α,β))
@test norm(jjt(c,α,β,α,β)-c,Inf) < 200eps()

println("Testing for Vector{Float32}")

c64 = rand(100)
c32 = map(Float32,c64)

cL64 = cjt(c64,0.,0.)
cL32 = cjt(c32,0.f0,0.f0)

@test norm(cL32-cL64,Inf) < 20eps(Float32)

println("Testing for Matrix of coefficients")

c = rand(100,100)
@test maxabs(jjt(c,α,β,α,β)-c) < 10000eps()

println("Testing Gaunt coefficients")

include("gaunttest.jl")

println("Testing BigFloat support of FFT and DCT methods")

include("fftBigFloattest.jl")

println("Testing equivalence of CXN and ASY methods")

for k in round(Int,logspace(1,4,20))
    r = randn(k)./√(1:k) # Proven 𝒪(√(log N)) error for ASY method.
    @test_approx_eq leg2cheb(r) cjt(r,0.,0.)
end

@test_approx_eq leg2chebu([1.0,2,3,4,5])  [0.546875,0.5,0.5390625,1.25,1.3671875]

c = randn(1000)./√(1:1000);

@test_approx_eq leg2cheb(cheb2leg(c)) c
@test_approx_eq cheb2leg(leg2cheb(c)) c

@test norm(jac2jac(c,0.,√2/2,-1/4,√2/2)-jjt(c,0.,√2/2,-1/4,√2/2),Inf) < 10length(c)*eps()

@test norm(ultra2ultra(ultra2ultra(c,.5,.75),.75,.5)-c,Inf) < 10length(c)*eps()


println("Testing (I)Padua Transforms and their inverse function property")
n=200
N=div((n+1)*(n+2),2)
v=rand(N)  #Length of v is the no. of Padua points
Pl=plan_paduatransform(v)
IPl=plan_ipaduatransform(v)
@test_approx_eq paduatransform(Pl,ipaduatransform(IPl,v)) v
@test_approx_eq ipaduatransform(IPl,paduatransform(Pl,v)) v

println("Testing runtimes for (I)Padua Transforms")
@time paduatransform(Pl,v)
@time ipaduatransform(IPl,v)

# println("Runtimes for Pre-planned (I)Padua Transforms")
# n=300
# v=rand(N)
# Plan=plan_paduatransform(v)
# IPlan=plan_ipaduatransform(v)
# @time paduatransform(Plan,v)
# @time ipaduatransform(IPlan,v)

println("Accuracy of 2d function interpolation at a point")
function trianglecfsmat{T}(cfs::AbstractVector{T})
    N=length(cfs)
    n=Int(cld(-3+sqrt(1+8N),2))
    @assert N==div((n+1)*(n+2),2)
    cfsmat=Array(T,n+2,n+1)
    cfsmat=fill!(cfsmat,0)
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
Interpolates a 2d function at a given point using 2d Chebyshev series.
"""
function paduaeval(f::Function,x::AbstractFloat,y::AbstractFloat,m::Integer)
    T=promote_type(typeof(x),typeof(y))
    M=div((m+1)*(m+2),2)
    pvals=Array(T,M)
    p=paduapoints(m)
    pvals=map!(f,p[:,1],p[:,2])
    plan=plan_paduatransform(pvals)
    coeffs=paduatransform(plan,pvals)
    cfs_mat=trianglecfsmat(coeffs)
    cfs_mat=view(cfs_mat,1:m+1,:)
    f_x=sum([cfs_mat[k,j]*cos((j-1)*acos(x))*cos((k-1)*acos(y)) for k=1:m+1, j=1:m+1])
    return f_x
end
f_xy = (x,y) -> x^2*y+x^3
g_xy = (x,y) ->cos(exp(2*x+y))*sin(y)
x=0.1;y=0.2
m=130
l=80
f_m=paduaeval(f_xy,x,y,m)
g_l=paduaeval(g_xy,x,y,l)
@test_approx_eq f_xy(x,y) f_m
@test_approx_eq g_xy(x,y) g_l

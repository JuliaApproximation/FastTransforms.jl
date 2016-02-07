using FastTransforms
using Base.Test

# Tests
n = 0:1000_000
λ = 0.123
@time FastTransforms.Cnλ(n,λ);
@time FastTransforms.Cnλ(n,λ);

x = linspace(0,20,81);
@test norm(FastTransforms.Cx(x)./FastTransforms.Cx(big(x))-1,Inf) < 2eps()

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

Nr = 10
v = zeros(Nr)
Na,Nb = 10,10
V = zeros(Na,Nb)

for N in round(Int,logspace(1,3,3))
    println("")
    println("N = ",N)
    println("")
    for αi=1:Na, βi=1:Nb
        α,β = -0.45+(αi-1)/10,-0.45+(βi-1)/10
        p1,p2 = plan_cjt(rand(N),α,β),plan_icjt(rand(N),α,β)
        for i=1:Nr
            c = rand(N)
            v[i] = log(norm(icjt(cjt(c,α,β,p1),α,β,p2)-c,Inf)/(20N^(1+2max(α,β))*eps()))
        end
        mean(v) > 2 && println("This is the mean log(||Error||_∞/Estimate) that broke the test: ",mean(v)," and the standard deviation: ",std(v))
        V[αi,βi] = mean(v)
    end
    println("This is the mean log(||Error||_∞/Estimate): ",mean(V)," and the standard deviation: ",std(V))
end


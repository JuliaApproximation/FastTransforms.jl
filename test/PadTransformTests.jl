include("PadTransform.jl")
using Base.Test

println("Testing Pad Transforms and their inverse function property")
n=100
N=div((n+1)*(n+2),2)
v=rand(N) # Length of v is the no. of Padua points
@test norm(padtransform(ipadtransform(v))-v)< 100eps()
@test norm(ipadtransform(padtransform(v))-v)< 100eps()
@test_approx_eq padtransform(ipadtransform(v)) v
@test_approx_eq ipadtransform(padtransform(v)) v

println("Testing runtimes for Pad Transforms")
@time padtransform(v)
@time ipadtransform(v)

println("Accuracy of 2d function interpolation at a point")
f = (x,y) -> x^2*y+x^3
g = (x,y) ->cos(exp(2*x+y))*sin(y)
x=0.1;y=0.2
m=20
l=80
f_x=padeval(f,x,y,m)
g_x=padeval(g,x,y,l)
@test_approx_eq f(x,y) f_x
@test_approx_eq g(x,y) g_x

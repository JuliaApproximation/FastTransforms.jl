using FastTransforms, LowRankApprox
using Base.Test

import FastTransforms: Butterfly

kernel = (x,y) -> exp(im*x*y)

function randfft(m,n,σ)
    x = (linspace(0,m-1,m)+σ*rand(m))
    y = (linspace(0,n-1,n)+σ*rand(n))*2π/n
    Complex{Float64}[kernel(x,y) for x in x, y in y]
end

function randnfft(m,n,σ)
    x = (linspace(0,m-1,m)+σ*randn(m))
    y = (linspace(0,n-1,n)+σ*randn(n))*2π/n
    Complex{Float64}[kernel(x,y) for x in x, y in y]
end

N = 10
A = Vector{Matrix{Complex{Float64}}}(N)
for n in 1:N
    A[n] = randnfft(2^n,2^n,0.0)
    println(n)
end

for n in 7:N
    println("N = ", n)
    @time B = Butterfly(A[n], n-5)
    b = rand(Complex{Float64},2^n)./(1:2^n)
    u = zero(b)
    @time uf = A[n]*b
    @time A_mul_B!(u, B, b)
    w = zero(b)
    @time Ac_mul_B!(w, B, u)
    scale!(inv(2^n), w)
    println(norm(u-uf)/2^n)
    println(norm(w-b))
    println(norm(w-A[n]\u))
end

N = 10
A = Vector{Matrix{Float64}}(N)
for n in 1:N
    A[n] = Float64[1/(i+j-1) for i = 1:2^n,j=1:2^n]
    println(n)
end

for n in 7:N
    println("N = ", n)
    @time B = Butterfly(A[n], n-5)
    b = rand(Float64,2^n)./(1:2^n)
    u = zero(b)
    @time uf = A[n]*b
    @time A_mul_B!(u, B, b)
    w = zero(b)
    @time At_mul_B!(w, B, b)
    println(norm(u-uf)/2^n)
    println(norm(w-A[n]'b))
    println(norm(u-w))
end


N = 10
A = Vector{Matrix{Complex{Float64}}}(N)
for n in 1:N
    A[n] = randnfft(2^n,2^n,0.1)
    println(n)
end

for n in 7:N
    println("N = ", n)
    @time B = Butterfly(A[n], n-5)
    b = rand(Complex{Float64},2^n)./(1:2^n)
    u = zero(b)
    @time uf = A[n]*b
    @time A_mul_B!(u, B, b)
    println(norm(u-uf)/2^n)
end


N = 12
A = Vector{Matrix{Complex{Float64}}}(N)
B = Vector{Butterfly{Complex{Float64}}}(N)
for n in 7:N
    A[n] = randnfft(2^n,2^n,0.1)
    @time B[n] = Butterfly(A[n], n-6)
    println(n)
end

for n in 7:N
    println("N = ", n)
    b = rand(Complex{Float64},2^n)./(1:2^n)
    uf = zero(b)
    u = zero(b)
    @time for k = 1:100
        A_mul_B!(uf, A[n], b)
    end
    @time for k = 1:100
        A_mul_B!(u, B[n], b)
    end
end
using FastTransforms, LowRankApprox, Compat
using Compat.Test, Compat.LinearAlgebra

if VERSION < v"0.7-"
    const mul! = Base.A_mul_B!
else
    Ac_mul_B!(C, A, B) = mul!(C, A', B)
end

import FastTransforms: Butterfly

@testset "Butterfly algorithm" begin
    kernel = (x,y) -> exp(im*x*y)

    function randfft(m,n,σ)
        x = (range(0,stop=m-1,length=m)+σ*rand(m))
        y = (range(0,stop=n-1,length=n)+σ*rand(n))*2π/n
        Complex{Float64}[kernel(x,y) for x in x, y in y]
    end

    function randnfft(m,n,σ)
        x = (range(0,stop=m-1,length=m)+σ*randn(m))
        y = (range(0,stop=n-1,length=n)+σ*randn(n))*2π/n
        Complex{Float64}[kernel(x,y) for x in x, y in y]
    end

    println("Testing the butterflied FFT")

    N = 10
    A = Vector{Matrix{Complex{Float64}}}(undef,N)
    for n in 1:N
        A[n] = randnfft(2^n,2^n,0.0)
    end

    for n in 7:N
        @time B = Butterfly(A[n], n-5)
        nb = size(A[n], 2)
        b = rand(Complex{Float64}, nb)./(1:nb)
        u = zero(b)
        @time uf = A[n]*b
        @time mul!(u, B, b)
        w = zero(b)
        @time Ac_mul_B!(w, B, u)
        lmul!(inv(2^n), w)
        println(norm(u-uf)/2^n)
        println(norm(w-b))
        println(norm(w-A[n]\u))
    end

    println("Testing the butterflied Hilbert matrix")

    N = 10
    A = Vector{Matrix{Float64}}(undef,N)
    for n in 1:N
        A[n] = Float64[1/(i+j-1) for i = 1:2^n+50,j=1:2^n+50]
    end

    for n in 7:N
        @time B = Butterfly(A[n], n-5)
        nb = size(A[n], 2)
        b = rand(Float64, nb)./(1:nb)
        u = zero(b)
        @time uf = A[n]*b
        @time mul!(u, B, b)
        w = zero(b)
        @time At_mul_B!(w, B, b)
        println(norm(u-uf)/nb)
        println(norm(w-A[n]'b))
        println(norm(u-w))
    end

    println("Testing the butterflied NUFFT")

    N = 10
    A = Vector{Matrix{Complex{Float64}}}(undef,N)
    B = Vector{Butterfly{Complex{Float64}}}(undef,N)
    for n in 7:N
        A[n] = randnfft(2^n+50,2^n+50,0.1)
        @time B[n] = Butterfly(A[n], n-6)
    end

    for n in 7:N
        nb = size(A[n], 2)
        b = rand(Complex{Float64}, nb)./(1:nb)
        uf = zero(b)
        u = zero(b)
        @time mul!(uf, A[n], b)
        @time mul!(u, B[n], b)
        println(norm(u-uf))
    end
end

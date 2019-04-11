using FastTransforms, Compat
using Compat.Test
if VERSION ≥ v"0.7-"
    using FFTW
end

@testset "BigFloat FFT and DCT" begin

    c = collect(range(-big(1.0),stop=1,length=16))
    @test norm(fft(c) - fft(Float64.(c))) < 3Float64(norm(c))*eps()
    @test norm(ifft(c) - ifft(Float64.(c))) < 3Float64(norm(c))*eps()

    c = collect(range(-big(1.0),stop=1.0,length=201))
    @test norm(ifft(fft(c))-c) < 200norm(c)eps(BigFloat)

    p = plan_dct(c)
    @test norm(FastTransforms.generic_dct(c) - p*c) == 0

    pi = plan_idct!(c)
    @test norm(pi*dct(c) - c) < 1000norm(c)*eps(BigFloat)

    @test norm(dct(c)-dct(map(Float64,c)),Inf) < 10eps()

    cc = cis.(c)
    @test norm(dct(cc)-dct(map(Complex{Float64},cc)),Inf) < 10eps()

    c = big.(rand(100)) + im*big.(rand(100))
    @test norm(dct(c)-dct(map(ComplexF64,c)),Inf) < 10eps()
    @test norm(idct(c)-idct(map(ComplexF64,c)),Inf) < 10eps()
    @test norm(idct(dct(c))-c,Inf) < 1000eps(BigFloat)
    @test norm(dct(idct(c))-c,Inf) < 1000eps(BigFloat)
end

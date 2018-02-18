using FastTransforms, Compat
using Compat.Test
if VERSION ≥ v"0.7-"
    using FFTW
end

@testset "BigFloat FFT and DCT" begin

    c = collect(linspace(-big(1.0),1,16))
    @test norm(fft(c) - fft(Float64.(c))) < 3Float64(norm(c))*eps()
    @test norm(ifft(c) - ifft(Float64.(c))) < 3Float64(norm(c))*eps()

    c = collect(linspace(-big(1.0),1.0,201))
    @test norm(ifft(fft(c))-c) < 200norm(c)eps(BigFloat)

    p = plan_dct(c)
    @test norm(dct(c) - p*c) == 0

    pi = plan_idct!(c)
    @test norm(pi*dct(c) - c) < 500norm(c)*eps(BigFloat)

    @test norm(dct(c)-dct(map(Float64,c)),Inf) < 10eps()

    cc = cis.(c)
    @test norm(dct(cc)-dct(map(Complex{Float64},cc)),Inf) < 10eps()
end

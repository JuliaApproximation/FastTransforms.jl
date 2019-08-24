using FastTransforms, FFTW, Test

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

    # Make sure we don't accidentally hijack any FFTW plans
    for T in (Float32, Float64)
        @test plan_fft(rand(BigFloat,10)) isa FastTransforms.DummyPlan
        @test plan_fft(rand(BigFloat,10), 1:1) isa FastTransforms.DummyPlan
        @test plan_fft(rand(Complex{BigFloat},10)) isa FastTransforms.DummyPlan
        @test plan_fft(rand(Complex{BigFloat},10), 1:1) isa FastTransforms.DummyPlan
        @test plan_fft!(rand(Complex{BigFloat},10)) isa FastTransforms.DummyPlan
        @test plan_fft!(rand(Complex{BigFloat},10), 1:1) isa FastTransforms.DummyPlan
        @test !( plan_fft(rand(T,10)) isa FastTransforms.DummyPlan )
        @test !( plan_fft(rand(T,10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_fft(rand(Complex{T},10)) isa FastTransforms.DummyPlan )
        @test !( plan_fft(rand(Complex{T},10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_fft!(rand(Complex{T},10)) isa FastTransforms.DummyPlan )
        @test !( plan_fft!(rand(Complex{T},10), 1:1) isa FastTransforms.DummyPlan )

        @test plan_ifft(rand(T,10)) isa FFTW.ScaledPlan
        @test plan_ifft(rand(T,10), 1:1) isa FFTW.ScaledPlan
        @test plan_ifft(rand(Complex{T},10)) isa FFTW.ScaledPlan
        @test plan_ifft(rand(Complex{T},10), 1:1) isa FFTW.ScaledPlan
        @test plan_ifft!(rand(Complex{T},10)) isa FFTW.ScaledPlan
        @test plan_ifft!(rand(Complex{T},10), 1:1) isa FFTW.ScaledPlan

        @test plan_bfft(rand(BigFloat,10)) isa FastTransforms.DummyPlan
        @test plan_bfft(rand(BigFloat,10), 1:1) isa FastTransforms.DummyPlan
        @test plan_bfft(rand(Complex{BigFloat},10)) isa FastTransforms.DummyPlan
        @test plan_bfft(rand(Complex{BigFloat},10), 1:1) isa FastTransforms.DummyPlan
        @test plan_bfft!(rand(Complex{BigFloat},10)) isa FastTransforms.DummyPlan
        @test plan_bfft!(rand(Complex{BigFloat},10), 1:1) isa FastTransforms.DummyPlan
        @test !( plan_bfft(rand(T,10)) isa FastTransforms.DummyPlan )
        @test !( plan_bfft(rand(T,10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_bfft(rand(Complex{T},10)) isa FastTransforms.DummyPlan )
        @test !( plan_bfft(rand(Complex{T},10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_bfft!(rand(Complex{T},10)) isa FastTransforms.DummyPlan )
        @test !( plan_bfft!(rand(Complex{T},10), 1:1) isa FastTransforms.DummyPlan )

        @test plan_dct(rand(BigFloat,10)) isa FastTransforms.DummyPlan
        @test plan_dct(rand(BigFloat,10), 1:1) isa FastTransforms.DummyPlan
        @test plan_dct(rand(Complex{BigFloat},10)) isa FastTransforms.DummyPlan
        @test plan_dct(rand(Complex{BigFloat},10), 1:1) isa FastTransforms.DummyPlan
        @test plan_dct!(rand(Complex{BigFloat},10)) isa FastTransforms.DummyPlan
        @test plan_dct!(rand(Complex{BigFloat},10), 1:1) isa FastTransforms.DummyPlan
        @test !( plan_dct(rand(T,10)) isa FastTransforms.DummyPlan )
        @test !( plan_dct(rand(T,10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_dct(rand(Complex{T},10)) isa FastTransforms.DummyPlan )
        @test !( plan_dct(rand(Complex{T},10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_dct!(rand(Complex{T},10)) isa FastTransforms.DummyPlan )
        @test !( plan_dct!(rand(Complex{T},10), 1:1) isa FastTransforms.DummyPlan )

        @test plan_idct(rand(BigFloat,10)) isa FastTransforms.DummyPlan
        @test plan_idct(rand(BigFloat,10), 1:1) isa FastTransforms.DummyPlan
        @test plan_idct(rand(Complex{BigFloat},10)) isa FastTransforms.DummyPlan
        @test plan_idct(rand(Complex{BigFloat},10), 1:1) isa FastTransforms.DummyPlan
        @test plan_idct!(rand(Complex{BigFloat},10)) isa FastTransforms.DummyPlan
        @test plan_idct!(rand(Complex{BigFloat},10), 1:1) isa FastTransforms.DummyPlan
        @test !( plan_idct(rand(T,10)) isa FastTransforms.DummyPlan )
        @test !( plan_idct(rand(T,10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_idct(rand(Complex{T},10)) isa FastTransforms.DummyPlan )
        @test !( plan_idct(rand(Complex{T},10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_idct!(rand(Complex{T},10)) isa FastTransforms.DummyPlan )
        @test !( plan_idct!(rand(Complex{T},10), 1:1) isa FastTransforms.DummyPlan )

        @test plan_rfft(rand(BigFloat,10)) isa FastTransforms.DummyPlan
        @test plan_rfft(rand(BigFloat,10), 1:1) isa FastTransforms.DummyPlan
        @test plan_brfft(rand(Complex{BigFloat},10), 19) isa FastTransforms.DummyPlan
        @test plan_brfft(rand(Complex{BigFloat},10), 19, 1:1) isa FastTransforms.DummyPlan
        @test !( plan_rfft(rand(T,10)) isa FastTransforms.DummyPlan )
        @test !( plan_rfft(rand(T,10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_brfft(rand(Complex{T},10), 19) isa FastTransforms.DummyPlan )
        @test !( plan_brfft(rand(Complex{T},10), 19, 1:1) isa FastTransforms.DummyPlan )

    end

end

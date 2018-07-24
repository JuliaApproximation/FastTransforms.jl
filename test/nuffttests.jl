using FastTransforms, Compat
using Compat.Test

if VERSION ≥ v"0.7-"
    using FFTW
    FFTW.set_num_threads(Base.Sys.CPU_THREADS)
else
    FFTW.set_num_threads(Base.Sys.CPU_CORES)
end

@testset "Nonuniform fast Fourier transforms" begin
    function nudft1(c::AbstractVector, ω::AbstractVector{T}) where {T<:AbstractFloat}
        # Nonuniform discrete Fourier transform of type I

        N = size(ω, 1)
        output = zero(c)
        for j = 1:N
        	output[j] = dot(exp.(2*T(π)*im*(j-1)/N*ω), c)
        end

        return output
    end

    function nudft2(c::AbstractVector, x::AbstractVector{T}) where {T<:AbstractFloat}
        # Nonuniform discrete Fourier transform of type II

        N = size(x, 1)
        output = zero(c)
        ω = collect(0:N-1)
        for j = 1:N
        	output[j] = dot(exp.(2*T(π)*im*x[j]*ω), c)
        end

        return output
    end

    function nudft3(c::AbstractVector, x::AbstractVector{T}, ω::AbstractVector{T}) where {T<:AbstractFloat}
        # Nonuniform discrete Fourier transform of type III

        N = size(x, 1)
        output = zero(c)
        for j = 1:N
            output[j] = dot(exp.(2*T(π)*im*x[j]*ω), c)
        end

        return output
    end

    N = round.([Int],10 .^ range(1,stop=3,length=10))

    for n in N, ϵ in (1e-4, 1e-8, 1e-12, eps(Float64))
        c = complex(rand(n))
        err_bnd = 500*ϵ*n*norm(c)

        ω = collect(0:n-1) + 0.25*rand(n)
        exact = nudft1(c, ω)
        fast = nufft1(c, ω, ϵ)
        @test norm(exact - fast, Inf) < err_bnd

        d = inufft1(fast, ω, ϵ)
        @test norm(c - d, Inf) < err_bnd

        x = (collect(0:n-1) + 0.25*rand(n))/n
        exact = nudft2(c, x)
        fast = nufft2(c, x, ϵ)
        @test norm(exact - fast, Inf) < err_bnd

        d = inufft2(fast, x, ϵ)
        @test norm(c - d, Inf) < err_bnd

        exact = nudft3(c, x, ω)
        fast = nufft3(c, x, ω, ϵ)
        @test norm(exact - fast, Inf) < err_bnd
    end

    # Check that if points/frequencies are indeed uniform, then it's equal to the fft.
    for n in (1000,), ϵ in (eps(Float64), 0.0)
        c = complex(rand(n))
        ω = collect(0.0:n-1)
        x = ω/n
        fftc = fft(c)
        if Base.Sys.WORD_SIZE == 64
            @test_skip norm(nufft1(c, ω, ϵ) - fftc) == 0 # skip because fftw3 seems to change this
            @test norm(nufft2(c, x, ϵ) - fftc) == 0
            @test_skip norm(nufft3(c, x, ω, ϵ) - fftc) == 0 # skip because fftw3 seems to change this
        end
        err_bnd = 500*eps(Float64)*norm(c)
        @test norm(nufft1(c, ω, ϵ) - fftc) < err_bnd
        @test norm(nufft2(c, x, ϵ) - fftc) < err_bnd
        @test norm(nufft3(c, x, ω, ϵ) - fftc) < err_bnd
    end

    function nudft1(C::Matrix{Complex{T}}, ω1::AbstractVector{T}, ω2::AbstractVector{T}) where {T<:AbstractFloat}
        # Nonuniform discrete Fourier transform of type I-I

        M, N = size(C)
        output = zero(C)
        @inbounds for j1 = 1:M, j2 = 1:N
            for k1 = 1:M, k2 = 1:N
                output[j1,j2] += exp(-2*T(π)*im*((j1-1)/M*ω1[k1]+(j2-1)/N*ω2[k2]))*C[k1,k2]
            end
        end
        return output
    end

    function nudft2(C::Matrix{Complex{T}}, x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat}
        # Nonuniform discrete Fourier transform of type II-II

        M, N = size(C)
        output = zero(C)
        @inbounds for j1 = 1:M, j2 = 1:N
            for k1 = 1:M, k2 = 1:N
                output[j1,j2] += exp(-2*T(π)*im*(x[j1]*(k1-1)+y[j2]*(k2-1)))*C[k1,k2]
            end
        end
        return output
    end

    N = round.([Int],10 .^ range(1,stop=1.7,length=5))

    for n in N, ϵ in (1e-4,1e-8,1e-12,eps(Float64))
        C = complex(rand(n,n))
        err_bnd = 500*ϵ*n*norm(C)

        x = (collect(0:n-1) + 0.25*rand(n))/n
        y = (collect(0:n-1) + 0.25*rand(n))/n
        ω1 = collect(0:n-1) + 0.25*rand(n)
        ω2 = collect(0:n-1) + 0.25*rand(n)

        exact = nudft1(C, ω1, ω2)
        fast = nufft1(C, ω1, ω2, ϵ)
        @test norm(exact - fast, Inf) < err_bnd

        exact = nudft2(C, x, y)
        fast = nufft2(C, x, y, ϵ)
        @test norm(exact - fast, Inf) < err_bnd
    end
end

# This file shows how to call `libfasttransforms` from Julia.

# Step 1: In this repository, `git clone -b v0.1.0 https://github.com/MikaelSlevinsky/FastTransforms.git`

# Step 2: use a version of gcc that supports OpenMP: on OS X, this means using a
# version of `gcc` from Homebrew, `brew install gcc`; on linux, `gcc-4.6` and up should work.
# `export CC=gcc-"the-right-version"`.

# Step 3: get the remaining dependencies: On OS X, either `brew install openblas`
# or change the Make.inc to use `BLAS = APPLEBLAS` instead of `BLAS = OPENBLAS`.
# Furthermore, `brew install fftw`. For linux, see the `Travis.yml` file.

# Step 4: run `make` and check the tests by running `./test_drivers 3 3 0`.
# All the errors should be roughly on the order of machine precision.

if VERSION < v"0.7-"
    using Base.Libdl
else
    using Libdl
end

import Base: unsafe_convert

const libfasttransforms = joinpath(dirname(@__DIR__), "FastTransforms", "libfasttransforms")

export plan_sph2fourier, plan_sph_synthesis, plan_sph_analysis, plan_sphv_synthesis, plan_sphv_analysis,
    sph2fourier!, fourier2sph!, sphv2fourier!, fourier2sphv!,
    sph_synthesis!, sph_analysis!, sphv_synthesis!, sphv_analysis!,
    plan_tri2cheb, plan_tri_synthesis, plan_tri_analysis,
    tri2cheb!, cheb2tri!, tri_synthesis!, tri_analysis!,
    plan_disk2cxf, plan_disk_synthesis, plan_disk_analysis,
    disk2cxf!, cxf2disk!, disk_synthesis!, disk_analysis!


struct ft_plan_struct end

abstract type HarmonicPlan{T} end

for P in (:SphericalHarmonicPlan, :TriangularHarmonicPlan, :DiskHarmonicPlan)
    @eval begin
        mutable struct $P{T} <: HarmonicPlan{T}
            plan::Ptr{ft_plan_struct}
            function $P{T}(plan::Ptr{ft_plan_struct}) where T
                p = new(plan)
                finalizer(destroy_harmonic_plan, p)
                p
            end
        end
    end
end

abstract type FTFFTWPlan{T} end

mutable struct SphereFFTWPlan{T} <: FTFFTWPlan{T}
    plan::Ptr{ft_plan_struct}
    function SphereFFTWPlan{T}(plan::Ptr{ft_plan_struct}) where T
        p = new(plan)
        finalizer(destroy_sphere_fftw_plan, p)
        p
    end
end

mutable struct TriangleFFTWPlan{T} <: FTFFTWPlan{T}
    plan::Ptr{ft_plan_struct}
    function TriangleFFTWPlan{T}(plan::Ptr{ft_plan_struct}) where T
        p = new(plan)
        finalizer(destroy_triangle_fftw_plan, p)
        p
    end
end

mutable struct DiskFFTWPlan{T} <: FTFFTWPlan{T}
    plan::Ptr{ft_plan_struct}
    function DiskFFTWPlan{T}(plan::Ptr{ft_plan_struct}) where T
        p = new(plan)
        finalizer(destroy_disk_fftw_plan, p)
        p
    end
end

unsafe_convert(::Type{Ptr{ft_plan_struct}}, P::HarmonicPlan{Float64}) = P.plan
unsafe_convert(::Type{Ptr{ft_plan_struct}}, P::FTFFTWPlan{Float64}) = P.plan

if Libdl.find_library(libfasttransforms) ≡ libfasttransforms
    set_num_threads(n::Int) = ccall((:ft_set_num_threads, libfasttransforms), Nothing, (Int, ), n)

    destroy_harmonic_plan(P::HarmonicPlan{Float64}) = ccall((:ft_destroy_harmonic_plan, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, ), P)

    function plan_sph2fourier(n::Int)
        plan = ccall((:ft_plan_sph2fourier, libfasttransforms), Ptr{ft_plan_struct}, (Int, ), n)
        return SphericalHarmonicPlan{Float64}(plan)
    end
    sph2fourier!(P::SphericalHarmonicPlan{Float64}, A::Matrix{Float64}) = ccall((:ft_execute_sph2fourier, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, Ptr{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))
    fourier2sph!(P::SphericalHarmonicPlan{Float64}, A::Matrix{Float64}) = ccall((:ft_execute_fourier2sph, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, Ptr{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))
    sphv2fourier!(P::SphericalHarmonicPlan{Float64}, A::Matrix{Float64}) = ccall((:ft_execute_sphv2fourier, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, Ptr{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))
    fourier2sphv!(P::SphericalHarmonicPlan{Float64}, A::Matrix{Float64}) = ccall((:ft_execute_fourier2sphv, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, Ptr{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))

    function plan_tri2cheb(n::Int, α::Float64, β::Float64, γ::Float64)
        plan = ccall((:ft_plan_tri2cheb, libfasttransforms), Ptr{ft_plan_struct}, (Int, Float64, Float64, Float64), n, α, β, γ)
        return TriangularHarmonicPlan{Float64}(plan)
    end
    tri2cheb!(P::TriangularHarmonicPlan{Float64}, A::Matrix{Float64}) = ccall((:ft_execute_tri2cheb, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, Ptr{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))
    cheb2tri!(P::TriangularHarmonicPlan{Float64}, A::Matrix{Float64}) = ccall((:ft_execute_cheb2tri, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, Ptr{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))

    function plan_disk2cxf(n::Int)
        plan = ccall((:ft_plan_disk2cxf, libfasttransforms), Ptr{ft_plan_struct}, (Int, ), n)
        return DiskHarmonicPlan{Float64}(plan)
    end
    disk2cxf!(P::DiskHarmonicPlan{Float64}, A::Matrix{Float64}) = ccall((:ft_execute_disk2cxf, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, Ptr{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))
    cxf2disk!(P::DiskHarmonicPlan{Float64}, A::Matrix{Float64}) = ccall((:ft_execute_cxf2disk, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, Ptr{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))


    destroy_sphere_fftw_plan(P::SphereFFTWPlan{Float64}) = ccall((:ft_destroy_sphere_fftw_plan, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, ), P)
    destroy_triangle_fftw_plan(P::TriangleFFTWPlan{Float64}) = ccall((:ft_destroy_triangle_fftw_plan, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, ), P)
    destroy_disk_fftw_plan(P::DiskFFTWPlan{Float64}) = ccall((:ft_destroy_disk_fftw_plan, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, ), P)

    function plan_sph_synthesis(n::Int, m::Int)
        plan = ccall((:ft_plan_sph_synthesis, libfasttransforms), Ptr{ft_plan_struct}, (Int, Int), n, m)
        SphereFFTWPlan{Float64}(plan)
    end
    function plan_sph_analysis(n::Int, m::Int)
        plan = ccall((:ft_plan_sph_analysis, libfasttransforms), Ptr{ft_plan_struct}, (Int, Int), n, m)
        SphereFFTWPlan{Float64}(plan)
    end
    function plan_sphv_synthesis(n::Int, m::Int)
        plan = ccall((:ft_plan_sphv_synthesis, libfasttransforms), Ptr{ft_plan_struct}, (Int, Int), n, m)
        SphereFFTWPlan{Float64}(plan)
    end
    function plan_sphv_analysis(n::Int, m::Int)
        plan = ccall((:ft_plan_sphv_analysis, libfasttransforms), Ptr{ft_plan_struct}, (Int, Int), n, m)
        SphereFFTWPlan{Float64}(plan)
    end

    sph_synthesis!(P::SphereFFTWPlan{Float64}, A::Matrix{Float64}) = ccall((:ft_execute_sph_synthesis, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, Ptr{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))
    sph_analysis!(P::SphereFFTWPlan{Float64}, A::Matrix{Float64}) = ccall((:ft_execute_sph_analysis, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, Ptr{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))
    sphv_synthesis!(P::SphereFFTWPlan{Float64}, A::Matrix{Float64}) = ccall((:ft_execute_sphv_synthesis, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, Ptr{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))
    sphv_analysis!(P::SphereFFTWPlan{Float64}, A::Matrix{Float64}) = ccall((:ft_execute_sphv_analysis, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, Ptr{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))

    function plan_tri_synthesis(n::Int, m::Int)
        plan = ccall((:ft_plan_tri_synthesis, libfasttransforms), Ptr{ft_plan_struct}, (Int, Int), n, m)
        TriangleFFTWPlan{Float64}(plan)
    end
    function plan_tri_analysis(n::Int, m::Int)
        plan = ccall((:ft_plan_tri_analysis, libfasttransforms), Ptr{ft_plan_struct}, (Int, Int), n, m)
        TriangleFFTWPlan{Float64}(plan)
    end
    tri_synthesis!(P::TriangleFFTWPlan{Float64}, A::Matrix{Float64}) = ccall((:ft_execute_tri_synthesis, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, Ptr{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))
    tri_analysis!(P::TriangleFFTWPlan{Float64}, A::Matrix{Float64}) = ccall((:ft_execute_tri_analysis, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, Ptr{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))

    function plan_disk_synthesis(n::Int, m::Int)
        plan = ccall((:ft_plan_disk_synthesis, libfasttransforms), Ptr{ft_plan_struct}, (Int, Int), n, m)
        DiskFFTWPlan{Float64}(plan)
    end
    function plan_disk_analysis(n::Int, m::Int)
        plan = ccall((:ft_plan_disk_analysis, libfasttransforms), Ptr{ft_plan_struct}, (Int, Int), n, m)
        DiskFFTWPlan{Float64}(plan)
    end
    disk_synthesis!(P::DiskFFTWPlan{Float64}, A::Matrix{Float64}) = ccall((:ft_execute_disk_synthesis, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, Ptr{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))
    disk_analysis!(P::DiskFFTWPlan{Float64}, A::Matrix{Float64}) = ccall((:ft_execute_disk_analysis, libfasttransforms), Nothing, (Ptr{ft_plan_struct}, Ptr{Float64}, Int, Int), P, A, size(A, 1), size(A, 2))
end

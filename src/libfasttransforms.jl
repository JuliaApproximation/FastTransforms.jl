ft_build_from_source = get(ENV, "FT_BUILD_FROM_SOURCE", "false")
if ft_build_from_source == "true"
    using Libdl
    const libfasttransforms = find_library("libfasttransforms", [joinpath(dirname(@__DIR__), "deps")])
    if libfasttransforms ≡ nothing || length(libfasttransforms) == 0
        error("FastTransforms is not properly installed. Please run Pkg.build(\"FastTransforms\") ",
              "and restart Julia.")
    end
else
    using FastTransforms_jll
end

function ft_fftw_plan_with_nthreads(n::Integer)
    ccall((:ft_fftw_plan_with_nthreads, libfasttransforms), Cvoid, (Cint, ), n)
end

function __init__()
    ccall((:ft_fftw_init_threads, libfasttransforms), Cint, ())
    ft_fftw_plan_with_nthreads(ceil(Int, Sys.CPU_THREADS/2))
end


"""
    mpfr_t <: AbstractFloat

A Julia struct that exactly matches `mpfr_t`.
"""
struct mpfr_t <: AbstractFloat
    prec::Clong
    sign::Cint
    exp::Clong
    d::Ptr{Limb}
end

"""
`BigFloat` is a mutable struct and there is no guarantee that each entry in
an `Array{BigFloat}` has unique pointers. For example, looking at the `Limb`s,

    Id = Matrix{BigFloat}(I, 3, 3)
    map(x->x.d, Id)

shows that the ones and the zeros all share the same pointers. If a C function
assumes unicity of each datum, then the array must be renewed with a `deepcopy`.
"""
function renew!(x::Array{BigFloat})
    for i in eachindex(x)
        @inbounds x[i] = deepcopy(x[i])
    end
    return x
end

set_num_threads(n::Integer) = ccall((:ft_set_num_threads, libfasttransforms), Cvoid, (Cint, ), n)

function horner!(c::Vector{Float64}, x::Vector{Float64}, f::Vector{Float64})
    @assert length(x) == length(f)
    ccall((:ft_horner, libfasttransforms), Cvoid, (Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Float64}), length(c), c, 1, length(x), x, f)
    f
end

function horner!(c::Vector{Float32}, x::Vector{Float32}, f::Vector{Float32})
    @assert length(x) == length(f)
    ccall((:ft_hornerf, libfasttransforms), Cvoid, (Cint, Ptr{Float32}, Cint, Cint, Ptr{Float32}, Ptr{Float32}), length(c), c, 1, length(x), x, f)
    f
end

function clenshaw!(c::Vector{Float64}, x::Vector{Float64}, f::Vector{Float64})
    @assert length(x) == length(f)
    ccall((:ft_clenshaw, libfasttransforms), Cvoid, (Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Float64}), length(c), c, 1, length(x), x, f)
    f
end

function clenshaw!(c::Vector{Float32}, x::Vector{Float32}, f::Vector{Float32})
    @assert length(x) == length(f)
    ccall((:ft_clenshawf, libfasttransforms), Cvoid, (Cint, Ptr{Float32}, Cint, Cint, Ptr{Float32}, Ptr{Float32}), length(c), c, 1, length(x), x, f)
    f
end

function clenshaw!(c::Vector{Float64}, A::Vector{Float64}, B::Vector{Float64}, C::Vector{Float64}, x::Vector{Float64}, phi0::Vector{Float64}, f::Vector{Float64})
    N = length(c)
    if length(A) < N || length(B) < N || length(C) < N
        throw(ArgumentError("A, B, C must contain at least $N entries"))
    end
    length(x) == length(phi0) == length(f) || throw(ArgumentError("Dimensions must match"))
    ccall((:ft_orthogonal_polynomial_clenshaw, libfasttransforms), Cvoid, (Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}), length(c), c, 1, A, B, C, length(x), x, phi0, f)
    f
end

function clenshaw!(c::Vector{Float32}, A::Vector{Float32}, B::Vector{Float32}, C::Vector{Float32}, x::Vector{Float32}, phi0::Vector{Float32}, f::Vector{Float32})
    @assert length(c) == length(A) == length(B) == length(C)-1
    @assert length(x) == length(phi0) == length(f)
    ccall((:ft_orthogonal_polynomial_clenshawf, libfasttransforms), Cvoid, (Cint, Ptr{Float32}, Cint, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Cint, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}), length(c), c, 1, A, B, C, length(x), x, phi0, f)
    f
end

const LEG2CHEB              = 0
const CHEB2LEG              = 1
const ULTRA2ULTRA           = 2
const JAC2JAC               = 3
const LAG2LAG               = 4
const JAC2ULTRA             = 5
const ULTRA2JAC             = 6
const JAC2CHEB              = 7
const CHEB2JAC              = 8
const ULTRA2CHEB            = 9
const CHEB2ULTRA           = 10
const SPHERE               = 11
const SPHEREV              = 12
const DISK                 = 13
const TRIANGLE             = 14
const TETRAHEDRON          = 15
const SPINSPHERE           = 16
const SPHERESYNTHESIS      = 17
const SPHEREANALYSIS       = 18
const SPHEREVSYNTHESIS     = 19
const SPHEREVANALYSIS      = 20
const DISKSYNTHESIS        = 21
const DISKANALYSIS         = 22
const TRIANGLESYNTHESIS    = 23
const TRIANGLEANALYSIS     = 24
const TETRAHEDRONSYNTHESIS = 25
const TETRAHEDRONANALYSIS  = 26
const SPINSPHERESYNTHESIS  = 27
const SPINSPHEREANALYSIS   = 28


let k2s = Dict(LEG2CHEB             => "Legendre--Chebyshev",
               CHEB2LEG             => "Chebyshev--Legendre",
               ULTRA2ULTRA          => "ultraspherical--ultraspherical",
               JAC2JAC              => "Jacobi--Jacobi",
               LAG2LAG              => "Laguerre--Laguerre",
               JAC2ULTRA            => "Jacobi--ultraspherical",
               ULTRA2JAC            => "ultraspherical--Jacobi",
               JAC2CHEB             => "Jacobi--Chebyshev",
               CHEB2JAC             => "Chebyshev--Jacobi",
               ULTRA2CHEB           => "ultraspherical--Chebyshev",
               CHEB2ULTRA           => "Chebyshev--ultraspherical",
               SPHERE               => "Spherical harmonic--Fourier",
               SPHEREV              => "Spherical vector field--Fourier",
               DISK                 => "Zernike--Chebyshev×Fourier",
               TRIANGLE             => "Proriol--Chebyshev²",
               TETRAHEDRON          => "Proriol--Chebyshev³",
               SPINSPHERE           => "Spin-weighted spherical harmonic--Fourier",
               SPHERESYNTHESIS      => "FFTW Fourier synthesis on the sphere",
               SPHEREANALYSIS       => "FFTW Fourier analysis on the sphere",
               SPHEREVSYNTHESIS     => "FFTW Fourier synthesis on the sphere (vector field)",
               SPHEREVANALYSIS      => "FFTW Fourier analysis on the sphere (vector field)",
               DISKSYNTHESIS        => "FFTW Chebyshev×Fourier synthesis on the disk",
               DISKANALYSIS         => "FFTW Chebyshev×Fourier analysis on the disk",
               TRIANGLESYNTHESIS    => "FFTW Chebyshev synthesis on the triangle",
               TRIANGLEANALYSIS     => "FFTW Chebyshev analysis on the triangle",
               TETRAHEDRONSYNTHESIS => "FFTW Chebyshev synthesis on the tetrahedron",
               TETRAHEDRONANALYSIS  => "FFTW Chebyshev analysis on the tetrahedron",
               SPINSPHERESYNTHESIS  => "FFTW Fourier synthesis on the sphere (spin-weighted)",
               SPINSPHEREANALYSIS   => "FFTW Fourier analysis on the sphere (spin-weighted)")
    global kind2string
    kind2string(k::Integer) = k2s[Int(k)]
end

struct ft_plan_struct end

mutable struct FTPlan{T, N, K}
    plan::Ptr{ft_plan_struct}
    n::Int
    l::Int
    m::Int
    function FTPlan{T, N, K}(plan::Ptr{ft_plan_struct}, n::Int) where {T, N, K}
        p = new(plan, n)
        finalizer(destroy_plan, p)
        p
    end
    function FTPlan{T, N, K}(plan::Ptr{ft_plan_struct}, n::Int, m::Int) where {T, N, K}
        p = new(plan, n, -1, m)
        finalizer(destroy_plan, p)
        p
    end
    function FTPlan{T, N, K}(plan::Ptr{ft_plan_struct}, n::Int, l::Int, m::Int) where {T, N, K}
        p = new(plan, n, l, m)
        finalizer(destroy_plan, p)
        p
    end
end

eltype(p::FTPlan{T}) where {T} = T
ndims(p::FTPlan{T, N}) where {T, N} = N
show(io::IO, p::FTPlan{T, 1, K}) where {T, K} = print(io, "FastTransforms ", kind2string(K), " plan for $(p.n)-element array of ", T)
show(io::IO, p::FTPlan{T, 2, SPHERE}) where T = print(io, "FastTransforms ", kind2string(SPHERE), " plan for $(p.n)×$(2p.n-1)-element array of ", T)
show(io::IO, p::FTPlan{T, 2, SPHEREV}) where T = print(io, "FastTransforms ", kind2string(SPHEREV), " plan for $(p.n)×$(2p.n-1)-element array of ", T)
show(io::IO, p::FTPlan{T, 2, DISK}) where T = print(io, "FastTransforms ", kind2string(DISK), " plan for $(p.n)×$(4p.n-3)-element array of ", T)
show(io::IO, p::FTPlan{T, 2, TRIANGLE}) where T = print(io, "FastTransforms ", kind2string(TRIANGLE), " plan for $(p.n)×$(p.n)-element array of ", T)
show(io::IO, p::FTPlan{T, 3, TETRAHEDRON}) where T = print(io, "FastTransforms ", kind2string(TETRAHEDRON), " plan for $(p.n)×$(p.n)×$(p.n)-element array of ", T)
show(io::IO, p::FTPlan{T, 2, SPINSPHERE}) where T = print(io, "FastTransforms ", kind2string(SPINSPHERE), " plan for $(p.n)×$(2p.n-1)-element array of ", T)
show(io::IO, p::FTPlan{T, 2, K}) where {T, K} = print(io, "FastTransforms plan for ", kind2string(K), " for $(p.n)×$(p.m)-element array of ", T)
show(io::IO, p::FTPlan{T, 3, K}) where {T, K} = print(io, "FastTransforms plan for ", kind2string(K), " for $(p.n)×$(p.l)×$(p.m)-element array of ", T)

function checksize(p::FTPlan{T}, x::Array{T}) where T
    if p.n != size(x, 1)
        throw(DimensionMismatch("FTPlan has dimensions $(p.n) × $(p.n), x has leading dimension $(size(x, 1))"))
    end
end

for K in (SPHERE, SPHEREV, DISK, SPINSPHERE)
    @eval function checksize(p::FTPlan{T, 2, $K}, x::Matrix{T}) where T
        if p.n != size(x, 1)
            throw(DimensionMismatch("FTPlan has dimensions $(p.n) × $(p.n), x has leading dimension $(size(x, 1))"))
        end
        if iseven(size(x, 2))
            throw(DimensionMismatch("This FTPlan only operates on arrays with an odd number of columns."))
        end
    end
end

unsafe_convert(::Type{Ptr{ft_plan_struct}}, p::FTPlan) = p.plan
unsafe_convert(::Type{Ptr{mpfr_t}}, p::FTPlan) = unsafe_convert(Ptr{mpfr_t}, p.plan)

destroy_plan(p::FTPlan{Float32, 1}) = ccall((:ft_destroy_tb_eigen_FMMf, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 1}) = ccall((:ft_destroy_tb_eigen_FMM, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{BigFloat, 1}) = ccall((:ft_mpfr_destroy_plan, libfasttransforms), Cvoid, (Ptr{mpfr_t}, Cint), p, p.n)
destroy_plan(p::FTPlan{Float64, 2}) = ccall((:ft_destroy_harmonic_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 3}) = ccall((:ft_destroy_tetrahedral_harmonic_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Complex{Float64}, 2, SPINSPHERE}) = ccall((:ft_destroy_spin_harmonic_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, SPHERESYNTHESIS}) = ccall((:ft_destroy_sphere_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, SPHEREANALYSIS}) = ccall((:ft_destroy_sphere_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, SPHEREVSYNTHESIS}) = ccall((:ft_destroy_sphere_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, SPHEREVANALYSIS}) = ccall((:ft_destroy_sphere_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, DISKSYNTHESIS}) = ccall((:ft_destroy_disk_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, DISKANALYSIS}) = ccall((:ft_destroy_disk_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, TRIANGLESYNTHESIS}) = ccall((:ft_destroy_triangle_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, TRIANGLEANALYSIS}) = ccall((:ft_destroy_triangle_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 3, TETRAHEDRONSYNTHESIS}) = ccall((:ft_destroy_tetrahedron_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 3, TETRAHEDRONANALYSIS}) = ccall((:ft_destroy_tetrahedron_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Complex{Float64}, 2, SPINSPHERESYNTHESIS}) = ccall((:ft_destroy_spinsphere_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Complex{Float64}, 2, SPINSPHEREANALYSIS}) = ccall((:ft_destroy_spinsphere_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)

struct AdjointFTPlan{T, S}
    parent::S
end

AdjointFTPlan(p::FTPlan) = AdjointFTPlan{eltype(p), typeof(p)}(p)

adjoint(p::FTPlan) = AdjointFTPlan(p)
adjoint(p::AdjointFTPlan) = p.parent

eltype(p::AdjointFTPlan{T, S}) where {T, S} = T
ndims(p::AdjointFTPlan{T, S}) where {T, S} = ndims(p.parent)
function show(io::IO, p::AdjointFTPlan{T, S}) where {T, S}
    print(io, "Adjoint ")
    show(io, p.parent)
end

checksize(p::AdjointFTPlan, x) = checksize(p.parent, x)

unsafe_convert(::Type{Ptr{ft_plan_struct}}, p::AdjointFTPlan{T, FTPlan{T, N, K}}) where {T, N, K} = unsafe_convert(Ptr{ft_plan_struct}, p.parent)
unsafe_convert(::Type{Ptr{mpfr_t}}, p::AdjointFTPlan{T, FTPlan{T, N, K}}) where {T, N, K} = unsafe_convert(Ptr{mpfr_t}, p.parent)

struct TransposeFTPlan{T, S}
    parent::S
end

TransposeFTPlan(p::FTPlan) = TransposeFTPlan{eltype(p), typeof(p)}(p)

transpose(p::FTPlan) = TransposeFTPlan(p)
transpose(p::TransposeFTPlan) = p.parent

eltype(p::TransposeFTPlan{T, S}) where {T, S} = T
ndims(p::TransposeFTPlan{T, S}) where {T, S} = ndims(p.parent)
function show(io::IO, p::TransposeFTPlan{T, S}) where {T, S}
    print(io, "Transpose ")
    show(io, p.parent)
end

checksize(p::TransposeFTPlan, x) = checksize(p.parent, x)

unsafe_convert(::Type{Ptr{ft_plan_struct}}, p::TransposeFTPlan{T, FTPlan{T, N, K}}) where {T, N, K} = unsafe_convert(Ptr{ft_plan_struct}, p.parent)
unsafe_convert(::Type{Ptr{mpfr_t}}, p::TransposeFTPlan{T, FTPlan{T, N, K}}) where {T, N, K} = unsafe_convert(Ptr{mpfr_t}, p.parent)

for f in (:leg2cheb, :cheb2leg, :ultra2ultra, :jac2jac,
          :lag2lag, :jac2ultra, :ultra2jac, :jac2cheb,
          :cheb2jac, :ultra2cheb, :cheb2ultra,
          :sph2fourier, :sphv2fourier, :disk2cxf, :tri2cheb, :tet2cheb)
    plan_f = Symbol("plan_", f)
    @eval begin
        $plan_f(x::AbstractArray{T}, y...; z...) where T = $plan_f(T, size(x, 1), y...; z...)
        $plan_f(::Type{Complex{T}}, y...; z...) where T <: Real = $plan_f(T, y...; z...)
        $f(x::AbstractArray, y...; z...) = $plan_f(x, y...; z...)*x
    end
end

for (f, plan_f) in ((:fourier2sph, :plan_sph2fourier), (:fourier2sphv, :plan_sphv2fourier),
                    (:cxf2disk2, :plan_disk2cxf), (:cheb2tri, :plan_tri2cheb),
                    (:cheb2tet, :plan_tet2cheb))
    @eval begin
        $f(x::AbstractArray, y...; z...) = $plan_f(x, y...; z...)\x
    end
end

plan_spinsph2fourier(x::AbstractArray{T}, y...; z...) where T = plan_spinsph2fourier(T, size(x, 1), y...; z...)
spinsph2fourier(x::AbstractArray, y...; z...) = plan_spinsph2fourier(x, y...; z...)*x
fourier2spinsph(x::AbstractArray, y...; z...) = plan_spinsph2fourier(x, y...; z...)\x

function plan_leg2cheb(::Type{Float32}, n::Integer; normleg::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_plan_legendre_to_chebyshevf, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint), normleg, normcheb, n)
    return FTPlan{Float32, 1, LEG2CHEB}(plan, n)
end

function plan_cheb2leg(::Type{Float32}, n::Integer; normcheb::Bool=false, normleg::Bool=false)
    plan = ccall((:ft_plan_chebyshev_to_legendref, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint), normcheb, normleg, n)
    return FTPlan{Float32, 1, CHEB2LEG}(plan, n)
end

function plan_ultra2ultra(::Type{Float32}, n::Integer, λ, μ; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_plan_ultraspherical_to_ultrasphericalf, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32, Float32), norm1, norm2, n, λ, μ)
    return FTPlan{Float32, 1, ULTRA2ULTRA}(plan, n)
end

function plan_jac2jac(::Type{Float32}, n::Integer, α, β, γ, δ; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_plan_jacobi_to_jacobif, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32, Float32, Float32, Float32), norm1, norm2, n, α, β, γ, δ)
    return FTPlan{Float32, 1, JAC2JAC}(plan, n)
end

function plan_lag2lag(::Type{Float32}, n::Integer, α, β; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_plan_laguerre_to_laguerref, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32, Float32), norm1, norm2, n, α, β)
    return FTPlan{Float32, 1, LAG2LAG}(plan, n)
end

function plan_jac2ultra(::Type{Float32}, n::Integer, α, β, λ; normjac::Bool=false, normultra::Bool=false)
    plan = ccall((:ft_plan_jacobi_to_ultrasphericalf, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32, Float32, Float32), normjac, normultra, n, α, β, λ)
    return FTPlan{Float32, 1, JAC2ULTRA}(plan, n)
end

function plan_ultra2jac(::Type{Float32}, n::Integer, λ, α, β; normultra::Bool=false, normjac::Bool=false)
    plan = ccall((:ft_plan_ultraspherical_to_jacobif, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32, Float32, Float32), normultra, normjac, n, λ, α, β)
    return FTPlan{Float32, 1, ULTRA2JAC}(plan, n)
end

function plan_jac2cheb(::Type{Float32}, n::Integer, α, β; normjac::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_plan_jacobi_to_chebyshevf, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32, Float32), normjac, normcheb, n, α, β)
    return FTPlan{Float32, 1, JAC2CHEB}(plan, n)
end

function plan_cheb2jac(::Type{Float32}, n::Integer, α, β; normcheb::Bool=false, normjac::Bool=false)
    plan = ccall((:ft_plan_chebyshev_to_jacobif, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32, Float32), normcheb, normjac, n, α, β)
    return FTPlan{Float32, 1, CHEB2JAC}(plan, n)
end

function plan_ultra2cheb(::Type{Float32}, n::Integer, λ; normultra::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_plan_ultraspherical_to_chebyshevf, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32), normultra, normcheb, n, λ)
    return FTPlan{Float32, 1, ULTRA2CHEB}(plan, n)
end

function plan_cheb2ultra(::Type{Float32}, n::Integer, λ; normcheb::Bool=false, normultra::Bool=false)
    plan = ccall((:ft_plan_chebyshev_to_ultrasphericalf, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32), normcheb, normultra, n, λ)
    return FTPlan{Float32, 1, CHEB2ULTRA}(plan, n)
end


function plan_leg2cheb(::Type{Float64}, n::Integer; normleg::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_plan_legendre_to_chebyshev, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint), normleg, normcheb, n)
    return FTPlan{Float64, 1, LEG2CHEB}(plan, n)
end

function plan_cheb2leg(::Type{Float64}, n::Integer; normcheb::Bool=false, normleg::Bool=false)
    plan = ccall((:ft_plan_chebyshev_to_legendre, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint), normcheb, normleg, n)
    return FTPlan{Float64, 1, CHEB2LEG}(plan, n)
end

function plan_ultra2ultra(::Type{Float64}, n::Integer, λ, μ; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_plan_ultraspherical_to_ultraspherical, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64, Float64), norm1, norm2, n, λ, μ)
    return FTPlan{Float64, 1, ULTRA2ULTRA}(plan, n)
end

function plan_jac2jac(::Type{Float64}, n::Integer, α, β, γ, δ; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_plan_jacobi_to_jacobi, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64, Float64, Float64, Float64), norm1, norm2, n, α, β, γ, δ)
    return FTPlan{Float64, 1, JAC2JAC}(plan, n)
end

function plan_lag2lag(::Type{Float64}, n::Integer, α, β; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_plan_laguerre_to_laguerre, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64, Float64), norm1, norm2, n, α, β)
    return FTPlan{Float64, 1, LAG2LAG}(plan, n)
end

function plan_jac2ultra(::Type{Float64}, n::Integer, α, β, λ; normjac::Bool=false, normultra::Bool=false)
    plan = ccall((:ft_plan_jacobi_to_ultraspherical, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64, Float64, Float64), normjac, normultra, n, α, β, λ)
    return FTPlan{Float64, 1, JAC2ULTRA}(plan, n)
end

function plan_ultra2jac(::Type{Float64}, n::Integer, λ, α, β; normultra::Bool=false, normjac::Bool=false)
    plan = ccall((:ft_plan_ultraspherical_to_jacobi, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64, Float64, Float64), normultra, normjac, n, λ, α, β)
    return FTPlan{Float64, 1, ULTRA2JAC}(plan, n)
end

function plan_jac2cheb(::Type{Float64}, n::Integer, α, β; normjac::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_plan_jacobi_to_chebyshev, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64, Float64), normjac, normcheb, n, α, β)
    return FTPlan{Float64, 1, JAC2CHEB}(plan, n)
end

function plan_cheb2jac(::Type{Float64}, n::Integer, α, β; normcheb::Bool=false, normjac::Bool=false)
    plan = ccall((:ft_plan_chebyshev_to_jacobi, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64, Float64), normcheb, normjac, n, α, β)
    return FTPlan{Float64, 1, CHEB2JAC}(plan, n)
end

function plan_ultra2cheb(::Type{Float64}, n::Integer, λ; normultra::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_plan_ultraspherical_to_chebyshev, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64), normultra, normcheb, n, λ)
    return FTPlan{Float64, 1, ULTRA2CHEB}(plan, n)
end

function plan_cheb2ultra(::Type{Float64}, n::Integer, λ; normcheb::Bool=false, normultra::Bool=false)
    plan = ccall((:ft_plan_chebyshev_to_ultraspherical, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64), normcheb, normultra, n, λ)
    return FTPlan{Float64, 1, CHEB2ULTRA}(plan, n)
end


function plan_leg2cheb(::Type{BigFloat}, n::Integer; normleg::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_mpfr_plan_legendre_to_chebyshev, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Clong, Int32), normleg, normcheb, n, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, LEG2CHEB}(plan, n)
end

function plan_cheb2leg(::Type{BigFloat}, n::Integer; normcheb::Bool=false, normleg::Bool=false)
    plan = ccall((:ft_mpfr_plan_chebyshev_to_legendre, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Clong, Int32), normcheb, normleg, n, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, CHEB2LEG}(plan, n)
end

function plan_ultra2ultra(::Type{BigFloat}, n::Integer, λ, μ; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_mpfr_plan_ultraspherical_to_ultraspherical, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Ref{BigFloat}, Clong, Int32), norm1, norm2, n, λ, μ, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, ULTRA2ULTRA}(plan, n)
end

function plan_jac2jac(::Type{BigFloat}, n::Integer, α, β, γ, δ; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_mpfr_plan_jacobi_to_jacobi, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Clong, Int32), norm1, norm2, n, α, β, γ, δ, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, JAC2JAC}(plan, n)
end

function plan_lag2lag(::Type{BigFloat}, n::Integer, α, β; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_mpfr_plan_laguerre_to_laguerre, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Ref{BigFloat}, Clong, Int32), norm1, norm2, n, α, β, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, LAG2LAG}(plan, n)
end

function plan_jac2ultra(::Type{BigFloat}, n::Integer, α, β, λ; normjac::Bool=false, normultra::Bool=false)
    plan = ccall((:ft_mpfr_plan_jacobi_to_ultraspherical, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Clong, Int32), normjac, normultra, n, α, β, λ, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, JAC2ULTRA}(plan, n)
end

function plan_ultra2jac(::Type{BigFloat}, n::Integer, λ, α, β; normultra::Bool=false, normjac::Bool=false)
    plan = ccall((:ft_mpfr_plan_ultraspherical_to_jacobi, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Clong, Int32), normultra, normjac, n, λ, α, β, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, ULTRA2JAC}(plan, n)
end

function plan_jac2cheb(::Type{BigFloat}, n::Integer, α, β; normjac::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_mpfr_plan_jacobi_to_chebyshev, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Ref{BigFloat}, Clong, Int32), normjac, normcheb, n, α, β, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, JAC2CHEB}(plan, n)
end

function plan_cheb2jac(::Type{BigFloat}, n::Integer, α, β; normcheb::Bool=false, normjac::Bool=false)
    plan = ccall((:ft_mpfr_plan_chebyshev_to_jacobi, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Ref{BigFloat}, Clong, Int32), normcheb, normjac, n, α, β, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, CHEB2JAC}(plan, n)
end

function plan_ultra2cheb(::Type{BigFloat}, n::Integer, λ; normultra::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_mpfr_plan_ultraspherical_to_chebyshev, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Clong, Int32), normultra, normcheb, n, λ, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, ULTRA2CHEB}(plan, n)
end

function plan_cheb2ultra(::Type{BigFloat}, n::Integer, λ; normcheb::Bool=false, normultra::Bool=false)
    plan = ccall((:ft_mpfr_plan_chebyshev_to_ultraspherical, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Clong, Int32), normcheb, normultra, n, λ, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, CHEB2ULTRA}(plan, n)
end


function plan_sph2fourier(::Type{Float64}, n::Integer)
    plan = ccall((:ft_plan_sph2fourier, libfasttransforms), Ptr{ft_plan_struct}, (Cint, ), n)
    return FTPlan{Float64, 2, SPHERE}(plan, n)
end

function plan_sphv2fourier(::Type{Float64}, n::Integer)
    plan = ccall((:ft_plan_sph2fourier, libfasttransforms), Ptr{ft_plan_struct}, (Cint, ), n)
    return FTPlan{Float64, 2, SPHEREV}(plan, n)
end

function plan_disk2cxf(::Type{Float64}, n::Integer)
    plan = ccall((:ft_plan_disk2cxf, libfasttransforms), Ptr{ft_plan_struct}, (Cint, ), n)
    return FTPlan{Float64, 2, DISK}(plan, n)
end

function plan_tri2cheb(::Type{Float64}, n::Integer, α, β, γ)
    plan = ccall((:ft_plan_tri2cheb, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Float64, Float64, Float64), n, α, β, γ)
    return FTPlan{Float64, 2, TRIANGLE}(plan, n)
end

function plan_tet2cheb(::Type{Float64}, n::Integer, α, β, γ, δ)
    plan = ccall((:ft_plan_tet2cheb, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Float64, Float64, Float64, Float64), n, α, β, γ, δ)
    return FTPlan{Float64, 3, TETRAHEDRON}(plan, n)
end

function plan_spinsph2fourier(::Type{Complex{Float64}}, n::Integer, s::Integer)
    plan = ccall((:ft_plan_spinsph2fourier, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint), n, s)
    return FTPlan{Complex{Float64}, 2, SPINSPHERE}(plan, n)
end

for (fJ, fC, fE, K) in ((:plan_sph_synthesis, :ft_plan_sph_synthesis, :ft_execute_sph_synthesis, SPHERESYNTHESIS),
                    (:plan_sph_analysis, :ft_plan_sph_analysis, :ft_execute_sph_analysis, SPHEREANALYSIS),
                    (:plan_sphv_synthesis, :ft_plan_sphv_synthesis, :ft_execute_sphv_synthesis, SPHEREVSYNTHESIS),
                    (:plan_sphv_analysis, :ft_plan_sphv_analysis, :ft_execute_sphv_analysis, SPHEREVANALYSIS),
                    (:plan_disk_synthesis, :ft_plan_disk_synthesis, :ft_execute_disk_synthesis, DISKSYNTHESIS),
                    (:plan_disk_analysis, :ft_plan_disk_analysis, :ft_execute_disk_analysis, DISKANALYSIS),
                    (:plan_tri_synthesis, :ft_plan_tri_synthesis, :ft_execute_tri_synthesis, TRIANGLESYNTHESIS),
                    (:plan_tri_analysis, :ft_plan_tri_analysis, :ft_execute_tri_analysis, TRIANGLEANALYSIS))
    @eval begin
        $fJ(x::Matrix{T}) where T = $fJ(T, size(x, 1), size(x, 2))
        $fJ(::Type{Complex{T}}, x...) where T <: Real = $fJ(T, x...)
        function $fJ(::Type{Float64}, n::Integer, m::Integer)
            plan = ccall(($(string(fC)), libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint), n, m)
            return FTPlan{Float64, 2, $K}(plan, n, m)
        end
        function lmul!(p::FTPlan{Float64, 2, $K}, x::Matrix{Float64})
            if p.n != size(x, 1) || p.m != size(x, 2)
                throw(DimensionMismatch("FTPlan has dimensions $(p.n) × $(p.m), x has dimensions $(size(x, 1)) × $(size(x, 2))"))
            end
            ccall(($(string(fE)), libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint), p, x, size(x, 1), size(x, 2))
            return x
        end
    end
end

plan_tet_synthesis(x::Array{T, 3}) where T = plan_tet_synthesis(T, size(x, 1), size(x, 2), size(x, 3))
plan_tet_synthesis(::Type{Complex{T}}, x...) where T <: Real = plan_tet_synthesis(T, x...)

function plan_tet_synthesis(::Type{Float64}, n::Integer, l::Integer, m::Integer)
    plan = ccall((:ft_plan_tet_synthesis, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint), n, l, m)
    return FTPlan{Float64, 3, TETRAHEDRONSYNTHESIS}(plan, n, l, m)
end

function lmul!(p::FTPlan{Float64, 3, TETRAHEDRONSYNTHESIS}, x::Array{Float64, 3})
    if p.n != size(x, 1) || p.l != size(x, 2) || p.m != size(x, 3)
        throw(DimensionMismatch("FTPlan has dimensions $(p.n) × $(p.l) × $(p.m), x has dimensions $(size(x, 1)) × $(size(x, 2)) × $(size(x, 3))"))
    end
    ccall((:ft_execute_tet_synthesis, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint, Cint), p, x, size(x, 1), size(x, 2), size(x, 3))
    return x
end

plan_tet_analysis(x::Array{T, 3}) where T = plan_tet_analysis(T, size(x, 1), size(x, 2), size(x, 3))
plan_tet_analysis(::Type{Complex{T}}, x...) where T <: Real = plan_tet_analysis(T, x...)

function plan_tet_analysis(::Type{Float64}, n::Integer, l::Integer, m::Integer)
    plan = ccall((:ft_plan_tet_analysis, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint), n, l, m)
    return FTPlan{Float64, 3, TETRAHEDRONANALYSIS}(plan, n, l, m)
end

function lmul!(p::FTPlan{Float64, 3, TETRAHEDRONANALYSIS}, x::Array{Float64, 3})
    if p.n != size(x, 1) || p.l != size(x, 2) || p.m != size(x, 3)
        throw(DimensionMismatch("FTPlan has dimensions $(p.n) × $(p.l) × $(p.m), x has dimensions $(size(x, 1)) × $(size(x, 2)) × $(size(x, 3))"))
    end
    ccall((:ft_execute_tet_analysis, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint, Cint), p, x, size(x, 1), size(x, 2), size(x, 3))
    return x
end

plan_spinsph_synthesis(x::Matrix{T}, s::Integer) where T = plan_spinsph_synthesis(T, size(x, 1), size(x, 2), s)

function plan_spinsph_synthesis(::Type{Complex{Float64}}, n::Integer, m::Integer, s::Integer)
    plan = ccall((:ft_plan_spinsph_synthesis, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint), n, m, s)
    return FTPlan{Complex{Float64}, 2, SPINSPHERESYNTHESIS}(plan, n, m)
end

function lmul!(p::FTPlan{Complex{Float64}, 2, SPINSPHERESYNTHESIS}, x::Matrix{Complex{Float64}})
    if p.n != size(x, 1) || p.m != size(x, 2)
        throw(DimensionMismatch("FTPlan has dimensions $(p.n) × $(p.m), x has dimensions $(size(x, 1)) × $(size(x, 2))"))
    end
    ccall((:ft_execute_spinsph_synthesis, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint), p, x, size(x, 1), size(x, 2))
    return x
end

plan_spinsph_analysis(x::Matrix{T}, s::Integer) where T = plan_spinsph_analysis(T, size(x, 1), size(x, 2), s)

function plan_spinsph_analysis(::Type{Complex{Float64}}, n::Integer, m::Integer, s::Integer)
    plan = ccall((:ft_plan_spinsph_analysis, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint), n, m, s)
    return FTPlan{Complex{Float64}, 2, SPINSPHEREANALYSIS}(plan, n, m)
end

function lmul!(p::FTPlan{Complex{Float64}, 2, SPINSPHEREANALYSIS}, x::Matrix{Complex{Float64}})
    if p.n != size(x, 1) || p.m != size(x, 2)
        throw(DimensionMismatch("FTPlan has dimensions $(p.n) × $(p.m), x has dimensions $(size(x, 1)) × $(size(x, 2))"))
    end
    ccall((:ft_execute_spinsph_analysis, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint), p, x, size(x, 1), size(x, 2))
    return x
end

*(p::FTPlan{T}, x::Array{T}) where T = lmul!(p, deepcopy(x))
*(p::AdjointFTPlan{T}, x::Array{T}) where T = lmul!(p, deepcopy(x))
*(p::TransposeFTPlan{T}, x::Array{T}) where T = lmul!(p, deepcopy(x))
\(p::FTPlan{T}, x::Array{T}) where T = ldiv!(p, deepcopy(x))
\(p::AdjointFTPlan{T}, x::Array{T}) where T = ldiv!(p, deepcopy(x))
\(p::TransposeFTPlan{T}, x::Array{T}) where T = ldiv!(p, deepcopy(x))

*(p::FTPlan{T, 1}, x::UniformScaling{S}) where {T, S} = lmul!(p, Matrix{promote_type(T, S)}(x, p.n, p.n))
*(p::AdjointFTPlan{T, FTPlan{T, 1, K}}, x::UniformScaling{S}) where {T, S, K} = lmul!(p, Matrix{promote_type(T, S)}(x, p.parent.n, p.parent.n))
*(p::TransposeFTPlan{T, FTPlan{T, 1, K}}, x::UniformScaling{S}) where {T, S, K} = lmul!(p, Matrix{promote_type(T, S)}(x, p.parent.n, p.parent.n))
\(p::FTPlan{T, 1}, x::UniformScaling{S}) where {T, S} = ldiv!(p, Matrix{promote_type(T, S)}(x, p.n, p.n))
\(p::AdjointFTPlan{T, FTPlan{T, 1, K}}, x::UniformScaling{S}) where {T, S, K} = ldiv!(p, Matrix{promote_type(T, S)}(x, p.parent.n, p.parent.n))
\(p::TransposeFTPlan{T, FTPlan{T, 1, K}}, x::UniformScaling{S}) where {T, S, K} = ldiv!(p, Matrix{promote_type(T, S)}(x, p.parent.n, p.parent.n))

for (fJ, fC, elty) in ((:lmul!, :ft_bfmvf, :Float32),
                       (:ldiv!, :ft_bfsvf, :Float32),
                       (:lmul!, :ft_bfmv , :Float64),
                       (:ldiv!, :ft_bfsv , :Float64))
    @eval begin
        function $fJ(p::FTPlan{$elty, 1}, x::Vector{$elty})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{$elty}), 'N', p, x)
            return x
        end
        function $fJ(p::AdjointFTPlan{$elty, FTPlan{$elty, 1, K}}, x::Vector{$elty}) where K
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{$elty}), 'T', p, x)
            return x
        end
        function $fJ(p::TransposeFTPlan{$elty, FTPlan{$elty, 1, K}}, x::Vector{$elty}) where K
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{$elty}), 'T', p, x)
            return x
        end
    end
end

for (fJ, fC) in ((:lmul!, :ft_mpfr_trmv_ptr),
                 (:ldiv!, :ft_mpfr_trsv_ptr))
    @eval begin
        function $fJ(p::FTPlan{BigFloat, 1}, x::Vector{BigFloat})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Ptr{mpfr_t}, Cint, Ptr{BigFloat}, Int32), 'N', p.n, p, p.n, renew!(x), Base.MPFR.ROUNDING_MODE[])
            return x
        end
        function $fJ(p::AdjointFTPlan{BigFloat, FTPlan{BigFloat, 1, K}}, x::Vector{BigFloat}) where K
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Ptr{mpfr_t}, Cint, Ptr{BigFloat}, Int32), 'T', p.parent.n, p, p.parent.n, renew!(x), Base.MPFR.ROUNDING_MODE[])
            return x
        end
        function $fJ(p::TransposeFTPlan{BigFloat, FTPlan{BigFloat, 1, K}}, x::Vector{BigFloat}) where K
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Ptr{mpfr_t}, Cint, Ptr{BigFloat}, Int32), 'T', p.parent.n, p, p.parent.n, renew!(x), Base.MPFR.ROUNDING_MODE[])
            return x
        end
    end
end

for (fJ, fC, elty) in ((:lmul!, :ft_bfmmf, :Float32),
                       (:ldiv!, :ft_bfsmf, :Float32),
                       (:lmul!, :ft_bfmm , :Float64),
                       (:ldiv!, :ft_bfsm , :Float64))
    @eval begin
        function $fJ(p::FTPlan{$elty, 1}, x::Matrix{$elty})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{$elty}, Cint, Cint), 'N', p, x, size(x, 1), size(x, 2))
            return x
        end
        function $fJ(p::AdjointFTPlan{$elty, FTPlan{$elty, 1, K}}, x::Matrix{$elty}) where K
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{$elty}, Cint, Cint), 'T', p, x, size(x, 1), size(x, 2))
            return x
        end
        function $fJ(p::TransposeFTPlan{$elty, FTPlan{$elty, 1, K}}, x::Matrix{$elty}) where K
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{$elty}, Cint, Cint), 'T', p, x, size(x, 1), size(x, 2))
            return x
        end
    end
end

for (fJ, fC) in ((:lmul!, :ft_mpfr_trmm_ptr),
                 (:ldiv!, :ft_mpfr_trsm_ptr))
    @eval begin
        function $fJ(p::FTPlan{BigFloat, 1}, x::Matrix{BigFloat})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Ptr{mpfr_t}, Cint, Ptr{BigFloat}, Cint, Cint, Int32), 'N', p.n, p, p.n, renew!(x), size(x, 1), size(x, 2), Base.MPFR.ROUNDING_MODE[])
            return x
        end
        function $fJ(p::AdjointFTPlan{BigFloat, FTPlan{BigFloat, 1, K}}, x::Matrix{BigFloat}) where K
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Ptr{mpfr_t}, Cint, Ptr{BigFloat}, Cint, Cint, Int32), 'T', p.parent.n, p, p.parent.n, renew!(x), size(x, 1), size(x, 2), Base.MPFR.ROUNDING_MODE[])
            return x
        end
        function $fJ(p::TransposeFTPlan{BigFloat, FTPlan{BigFloat, 1, K}}, x::Matrix{BigFloat}) where K
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Ptr{mpfr_t}, Cint, Ptr{BigFloat}, Cint, Cint, Int32), 'T', p.parent.n, p, p.parent.n, renew!(x), size(x, 1), size(x, 2), Base.MPFR.ROUNDING_MODE[])
            return x
        end
    end
end

for (fJ, fC, K) in ((:lmul!, :ft_execute_sph2fourier, SPHERE),
                    (:ldiv!, :ft_execute_fourier2sph, SPHERE),
                    (:lmul!, :ft_execute_sphv2fourier, SPHEREV),
                    (:ldiv!, :ft_execute_fourier2sphv, SPHEREV),
                    (:lmul!, :ft_execute_disk2cxf, DISK),
                    (:ldiv!, :ft_execute_cxf2disk, DISK),
                    (:lmul!, :ft_execute_tri2cheb, TRIANGLE),
                    (:ldiv!, :ft_execute_cheb2tri, TRIANGLE))
    @eval begin
        function $fJ(p::FTPlan{Float64, 2, $K}, x::Matrix{Float64})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint), p, x, size(x, 1), size(x, 2))
            return x
        end
    end
end

function lmul!(p::FTPlan{Float64, 3, TETRAHEDRON}, x::Array{Float64, 3})
    checksize(p, x)
    ccall((:ft_execute_tet2cheb, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint, Cint), p, x, size(x, 1), size(x, 2), size(x, 3))
    return x
end

function ldiv!(p::FTPlan{Float64, 3, TETRAHEDRON}, x::Array{Float64, 3})
    checksize(p, x)
    ccall((:ft_execute_cheb2tet, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint, Cint), p, x, size(x, 1), size(x, 2), size(x, 3))
    return x
end

function lmul!(p::FTPlan{Complex{Float64}, 2, SPINSPHERE}, x::Matrix{Complex{Float64}})
    checksize(p, x)
    ccall((:ft_execute_spinsph2fourier, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, Ptr{Complex{Float64}}, Cint, Cint), p, x, size(x, 1), size(x, 2))
    return x
end

function ldiv!(p::FTPlan{Complex{Float64}, 2, SPINSPHERE}, x::Matrix{Complex{Float64}})
    checksize(p, x)
    ccall((:ft_execute_fourier2spinsph, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, Ptr{Complex{Float64}}, Cint, Cint), p, x, size(x, 1), size(x, 2))
    return x
end

*(p::FTPlan{T}, x::Array{Complex{T}}) where T = lmul!(p, deepcopy(x))
*(p::AdjointFTPlan{T}, x::Array{Complex{T}}) where T = lmul!(p, deepcopy(x))
*(p::TransposeFTPlan{T}, x::Array{Complex{T}}) where T = lmul!(p, deepcopy(x))
\(p::FTPlan{T}, x::Array{Complex{T}}) where T = ldiv!(p, deepcopy(x))
\(p::AdjointFTPlan{T}, x::Array{Complex{T}}) where T = ldiv!(p, deepcopy(x))
\(p::TransposeFTPlan{T}, x::Array{Complex{T}}) where T = ldiv!(p, deepcopy(x))

for fJ in (:lmul!, :ldiv!)
    @eval begin
        function $fJ(p::FTPlan{T}, x::AbstractArray{Complex{T}}) where T
            x .= complex.($fJ(p, real(x)), $fJ(p, imag(x)))
            return x
        end
        function $fJ(p::AdjointFTPlan{T, FTPlan{T, N, K}}, x::AbstractArray{Complex{T}}) where {T, N, K}
            x .= complex.($fJ(p, real(x)), $fJ(p, imag(x)))
            return x
        end
        function $fJ(p::TransposeFTPlan{T, FTPlan{T, N, K}}, x::AbstractArray{Complex{T}}) where {T, N, K}
            x .= complex.($fJ(p, real(x)), $fJ(p, imag(x)))
            return x
        end
    end
end
